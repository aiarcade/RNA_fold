
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

class RNA_Dataset(Dataset):
    def __init__(self,df, experiment):
       
        df[df['experiment_type'] == experiment]

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
        return seq,reactivity

    def encode_rna_sequence(self,sequence):
        nucleotide_mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'U': 1.0}
        mapped_seq=[nucleotide_mapping[nt] for nt in sequence]
        encoded_sequence = np.array(mapped_seq)
        original_tensor=torch.Tensor(encoded_sequence)
        #padding_size = max(0, SEQ_LEN - original_tensor.size(0))
        #padded_tensor = torch.nn.functional.pad(original_tensor, (0, padding_size), mode='constant', value=0)
        return original_tensor 


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