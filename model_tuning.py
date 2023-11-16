import argparse
import os
from typing import List
from typing import Optional


from packaging import version
import torch
from torch import nn
from torch import optim

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from models import *

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
EPOCHS = 2

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


class RNADataModule(pl.LightningDataModule):
    def __init__(self, experiment: str, batch_size: int):
        super().__init__()
        self.experiment =  experiment
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        data=pd.read_parquet('train_data.parquet', engine='fastparquet')
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



def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_rnnlayers = trial.suggest_int("n_layers",8,100)
    h_size   =  trial.suggest_int(f'h_units',8,64)
    l_size =  trial.suggest_int(f'l_units',8,64)
    lr = trial.suggest_float('learning_rate', 1e-5, 1000,log=True)
    model = SimpleRNN(input_size=1, hidden_size=h_size,output_size=1,n_rnn_layers=n_rnnlayers,linear_size=l_size,learning_rate=lr)
    datamodule = RNADataModule(experiment="DMS_MaP", batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(hidden_size=h_size,n_rnn_layers=n_rnnlayers,linear_size=l_size,learning_rate=lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

