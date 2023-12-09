from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup,T5Config
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pandas as pd
import os, gc
import numpy as np

import os
from typing import List
from typing import Optional
from settings import *
import math

class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=42, fold=0, nfolds=4, 
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 457
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        
        #split = list(KFold(n_splits=nfolds, random_state=seed, 
        #        shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.reset_index(drop=True)
        df_DMS = df_DMS.reset_index(drop=True)
        
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        
        react_2A3=self.react_2A3[idx]
        react_2A3= np.pad(react_2A3,(0,self.Lmax-len(react_2A3)))

        react_DMS=self.react_DMS[idx]
        react_DMS= np.pad(react_DMS,(0,self.Lmax-len(react_DMS)))

        react = torch.from_numpy(np.stack([react_2A3,
                                           react_DMS],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])

        #print("loader",react.shape)
        
        return {'seq':seq, 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}

class RNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_file:str):
        super().__init__()
        self.batch_size = batch_size
        self.file=data_file

    def setup(self, stage: Optional[str] = None) -> None:
        data=pd.read_parquet(self.file, engine='fastparquet')
        dataset=RNA_Dataset(data)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=24
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


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=32768):#10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RNAT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.emb = nn.Embedding(4, 512)
        self.pos_enc = SinusoidalPosEmb(512)
        config=T5Config()
        self.model = T5ForConditionalGeneration(config)
        self.save_hyperparameters()

    def forward(self, x0): 
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        #x = self.emb(x)
        #x = x + pos    
        outputs = self.model(input_ids=x, attention_mask=mask)
        return outputs
    
    def common_step(self, batch, batch_idx):
        x,y=batch
        outputs = self(x)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss


    def configure_optimizers(self):
        # # create optimizer
        # optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # # create learning rate scheduler
        # num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        # lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=self.hparams.warmup_steps,
        #                                             num_training_steps=num_train_optimization_steps),
        #                 'name': 'learning_rate',
        #                 'interval':'step',
        #                 'frequency': 1}
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
        
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader


# model=RNAT5()
# datamodule = RNADataModule(batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_DATASET_FILE)
# early_stop_callback = EarlyStopping(
#     monitor='validation_loss',
#     patience=3,
#     strict=False,
#     verbose=False,
#     mode='min'
# )
# lr_monitor = LearningRateMonitor(logging_interval='step')

# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     monitor='validation_loss',
#     mode='min',
#     save_top_k=1,
#     save_last=True,
#     filename='t5-model-epoch_{epoch:02d}_val_loss_{val_loss:.2f}',
#     every_n_epochs=1,
#     dirpath=MODEL_DIR_PREFIX
# )

# trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback,early_stop_callback, lr_monitor],
#             accelerator=ACCELERATION, devices=DEVICES, 
#             strategy="ddp")

# trainer.fit(model,datamodule=datamodule)
# import torch
# from transformers import AutoTokenizer, OpenAIGPTLMHeadModel
# data=pd.read_parquet(TRAIN_DATASET_FILE, engine='fastparquet')

# dataset=RNA_Dataset(data)

# tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
# model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

# inputs = tokenizer(dataset[0][0]['seq'], return_tensors="pt")
# print(inputs)
# outputs = model(**inputs, labels=inputs["input_ids"])
# print(outputs.loss)
#loss = outputs.loss
#logits = outputs.logits