
import os
from typing import List
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import glob
from torchvision import transforms
from arnie.bpps import bpps
class RNA_Dataset(Dataset):
    def __init__(self,df, experiment):
       
        df=df[df['experiment_type'] == experiment]

        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','sequence_id','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        self.df=df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
        seq_len=len(row['sequence'])
        selected_reactivities=row['reactivity'][0:seq_len]
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        #print(idx,seq.shape,reactivity.shape)
        reactivity=self.pad_tensor(reactivity,207)
        return seq,reactivity

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        padding_size = max(0, 207 - original_tensor.size(0))
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return padded_tensor 

    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor


class RNA_TestDataset(Dataset):
    def __init__(self,df):
        self.df=df
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
 
        return seq,[row['id_min'],row['id_max']]

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        return original_tensor 


class RNADataModule(pl.LightningDataModule):
    def __init__(self, experiment: str, batch_size: int, data_file: str = "subset_data.parquet"):
        super().__init__()
        self.experiment =  experiment
        self.batch_size = batch_size
        self.file=data_file

    def setup(self, stage: Optional[str] = None) -> None:
        data=pd.read_parquet(self.file, engine='fastparquet')
        dataset=RNA_Dataset(data,self.experiment)
        train_size = int(0.8 * len(dataset))
        val_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - val_size
        self.no_workers=7
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,collate_fn=self.custom_collate_fn,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,collate_fn=self.custom_collate_fn,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,collate_fn=self.custom_collate_fn,pin_memory=True)
    
    def custom_collate_fn(self,data):
        inputs=[]
        labels=[]
        for x,y in data:
            inputs.append(x)
            labels.append(y)
        input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        return input_tensors,label_tensors


class StructureProbDataset(Dataset):
    def __init__(self,src_dir):
        files_pattern = os.path.join(src_dir, '*.npz')
        self.file_list = glob.glob(files_pattern)
        self.transforms_seq=transforms.Compose([transforms.Resize([177,177])])

    def __len__(self):
        return len(self.file_list)  
    
    def __getitem__(self, idx):
        
        src_file=self.file_list[idx]
        row=np.load(src_file,allow_pickle=True)
        seq=torch.Tensor(row['seq'])
        selected_reactivities=row['reactivity']
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        #print(idx,seq.shape,reactivity.shape)
        seq=self.transforms_seq(seq.unsqueeze(0))
        batch=seq,self.pad_tensor(reactivity,177)
        #print(batch[0].shape,batch[1].shape)
        return batch

    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor


class ProbDataModule(pl.LightningDataModule):
    def __init__(self, src_dir: str, batch_size: int):
        super().__init__()
        self.src_dir= src_dir
        self.batch_size = batch_size


    def setup(self, stage: Optional[str] = None) -> None:
        dataset=StructureProbDataset(self.src_dir)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        self.no_workers=1
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def custom_collate_fn(self,data):
        # inputs=[]
        # labels=[]
        # for x,y in data:
        #     inputs.append(x)
        #     labels.append(y)
        # input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        # label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        # return input_tensors,label_tensors
        return data




class StructureProbTestDataset(Dataset):
    def __init__(self,src_file):
        
        self.transforms_seq=transforms.Compose([transforms.Resize([177,177])])
        self.src_file=src_file
        self.df = pd.read_csv(self.src_file)

    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        row=self.df.loc[idx]
        ids=np.array([row['id_min'],row['id_max']])
        seq=self.encode_rna_sequence(row['sequence'])
        seq=torch.Tensor(seq)
        seq=self.transforms_seq(seq.unsqueeze(0))
        
        return seq,ids,idx



    def encode_rna_sequence(self,sequence):
        return bpps(sequence, package='eternafold')

       
class StructureProbDatasetWithFixed500(Dataset):
    def __init__(self,src_dir):
        files_pattern = os.path.join(src_dir, '*.npz')
        self.file_list = glob.glob(files_pattern)
        self.transforms_seq=transforms.Compose([transforms.Resize([500,500],antialias=False)])

    def __len__(self):
        return int(len(self.file_list)) 
    
    def __getitem__(self, idx):
        
        src_file=self.file_list[idx]
        row=np.load(src_file,allow_pickle=True)
        seq=torch.Tensor(row['seq'])
        selected_reactivities=row['reactivity']
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        #print(idx,seq.shape,reactivity.shape)
        seq=self.transforms_seq(seq.unsqueeze(0))
        batch=seq,self.pad_tensor(reactivity,500)
        #print(batch[0].shape,batch[1].shape)
        return batch

    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor


class ProbDataModuleWithFixed500(pl.LightningDataModule):
    def __init__(self, src_dir: str, batch_size: int):
        super().__init__()
        self.src_dir= src_dir
        self.batch_size = batch_size


    def setup(self, stage: Optional[str] = None) -> None:
        dataset=StructureProbDatasetWithFixed500(self.src_dir)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        self.no_workers=63
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def custom_collate_fn(self,data):
        # inputs=[]
        # labels=[]
        # for x,y in data:
        #     inputs.append(x)
        #     labels.append(y)
        # input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        # label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        # return input_tensors,label_tensors
        return data


class StructureProbTestDataset500(Dataset):
    def __init__(self,src_dir):
        
        files = os.listdir(src_dir)
        numeric_parts=[]
        for file in files:
            numeric_parts.append(int(file.replace(".npz","")))
        self.file_list = sorted(numeric_parts)

        self.transforms_seq=transforms.Compose([transforms.Resize([500,500],antialias=False)])
        self.src_dir=src_dir

    def __len__(self):
        return len(self.file_list)  
    
    def __getitem__(self, idx):
        src_file= self.src_dir+str(self.file_list[idx])+'.npz'
        row=np.load(src_file,allow_pickle=True)
        seq=torch.Tensor(row['seq'])
        ids=row['ids']
        #print(idx,seq.shape,reactivity.shape)
        seq=self.transforms_seq(seq.unsqueeze(0))
        
        #print(batch[0].shape,batch[1].shape)
        return seq,ids

class RNA_ImageDataset(Dataset):
    def __init__(self,df, experiment):
       
        df=df[df['experiment_type'] == experiment]

        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','sequence_id','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        self.df=df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
        seq_len=len(row['sequence'])
        selected_reactivities=row['reactivity'][0:seq_len]
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        reactivity=self.pad_tensor(reactivity,500)
        #print(idx,seq.shape,reactivity.shape)
        return seq.unsqueeze(0),reactivity

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        current_length=len(original_tensor)
        pad_end = max(0, 500 - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, pad_end), value=0)
        expanded_tensor = torch.unsqueeze(padded_tensor , dim=1)
        repeated_tensor = expanded_tensor.repeat(1, 500)
        #padding_size = max(0, SEQ_LEN - original_tensor.size(0))
        #padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return repeated_tensor
    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor

class DataModuleRNAImageDataset(pl.LightningDataModule):
    def __init__(self, src_file: str, experiment :str,batch_size: int):
        super().__init__()
        self.src_file= src_file
        self.batch_size = batch_size
        self.experiment=experiment
        self.df=pd.read_parquet(self.src_file)


    def setup(self, stage: Optional[str] = None) -> None:
        dataset=RNA_ImageDataset(self.df,self.experiment)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        self.no_workers=63
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def custom_collate_fn(self,data):
        # inputs=[]
        # labels=[]
        # for x,y in data:
        #     inputs.append(x)
        #     labels.append(y)
        # input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        # label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        # return input_tensors,label_tensors
        return data


class RNA_NNDataset(Dataset):
    def __init__(self,df, experiment):
       
        df=df[df['experiment_type'] == experiment]

        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','sequence_id','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        self.df=df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
        seq_len=len(row['sequence'])
        selected_reactivities=row['reactivity'][0:seq_len]
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        #print(idx,seq.shape,reactivity.shape)
        reactivity=self.pad_tensor(reactivity,480)
        #print(seq.shape,reactivity.shape)
        return seq.unsqueeze(0),reactivity

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        padding_size = max(0, 480 - original_tensor.size(0))
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return padded_tensor 

    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor

class RNANNDataModule(pl.LightningDataModule):
    def __init__(self, experiment: str, batch_size: int, data_file: str = "subset_data.parquet"):
        super().__init__()
        self.experiment =  experiment
        self.batch_size = batch_size
        self.file=data_file

    def setup(self, stage: Optional[str] = None) -> None:
        data=pd.read_parquet(self.file, engine='fastparquet')
        dataset=RNA_NNDataset(data,self.experiment)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=7
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def custom_collate_fn(self,data):
        inputs=[]
        labels=[]
        for x,y in data:
            inputs.append(x)
            labels.append(y)
        input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        return input_tensors,label_tensors

class RNA_NNTestDataset(Dataset):
    def __init__(self,df):
        self.df=df
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
 
        return seq,[row['id_min'],row['id_max']]

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        padding_size = max(0, 480 - original_tensor.size(0))
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return  padded_tensor 


class RNA_TRDataset(Dataset):
    def __init__(self,df, experiment):
       
        df=df[df['experiment_type'] == experiment]

        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','sequence_id','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        self.df=df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
        seq_len=len(row['sequence'])
        selected_reactivities=row['reactivity'][0:seq_len]
        reactivity = torch.Tensor(selected_reactivities)
        reactivity[reactivity.isnan()] = 0.0
        reactivity=torch.clamp(reactivity, min=0)
        #print(idx,seq.shape,reactivity.shape)
        reactivity=self.pad_tensor(reactivity,480)
        #print(seq.shape,reactivity.shape)
        return seq.unsqueeze(0),reactivity.unsqueeze(0)

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        padding_size = max(0, 480 - original_tensor.size(0))
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return padded_tensor 

    def pad_tensor(self,input_tensor, desired_length):
        current_length = len(input_tensor)
        if current_length>desired_length:
            return input_tensor[:desired_length]
        
        # Calculate the amount of padding needed at the end
        pad_end = max(0, desired_length - current_length)
        
        # Use torch.nn.functional.pad to pad the tensor at the end
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_end), value=0)
        
        return padded_tensor

class RNATRDataModule(pl.LightningDataModule):
    def __init__(self, experiment: str, batch_size: int, data_file: str = "subset_data.parquet"):
        super().__init__()
        self.experiment =  experiment
        self.batch_size = batch_size
        self.file=data_file

    def setup(self, stage: Optional[str] = None) -> None:
        data=pd.read_parquet(self.file, engine='fastparquet')
        dataset=RNA_TRDataset(data,self.experiment)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=7
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def custom_collate_fn(self,data):
        inputs=[]
        labels=[]
        for x,y in data:
            inputs.append(x)
            labels.append(y)
        input_tensors = pad_sequence(inputs, batch_first=True, padding_value=0)
        label_tensors = pad_sequence(labels, batch_first=True, padding_value=0)
        return input_tensors,label_tensors


class RNA_TFTestDataset(Dataset):
    def __init__(self,df):
        self.df=df
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        row=self.df.loc[idx]
        seq=self.encode_rna_sequence(row['sequence'])
 
        return seq,row['id_min'],row['id_max']

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        padding_size = max(0, 480 - original_tensor.size(0))
        padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return  padded_tensor 
