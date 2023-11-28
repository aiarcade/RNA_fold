import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
from settings import *
import torch.optim as optim
import pytorch_lightning as pl
import os
from typing import List
from typing import Optional
import sys
import math
from tqdm import tqdm
from lightning.pytorch.strategies import DeepSpeedStrategy

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
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        
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
        
        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}
    
class RNA_TestDataset(Dataset):
    def __init__(self, df, mode='train', seed=42, fold=0, nfolds=4, 
                 mask_only=False,device="cuda:3"):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 457
        df['L'] = df.sequence.apply(len)
        self.device=device

        
        #split = list(KFold(n_splits=nfolds, random_state=seed, 
        #        shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
       
        
        
        self.seq = df['sequence'].values
        self.L = df['L'].values
        self.id_min=df['id_min']
        self.id_max=df['id_max']

        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        id_min=self.id_min[idx]
        id_max=self.id_max[idx]
        
               
        return {'seq':torch.from_numpy(seq).to(self.device), 'mask':mask.to(self.device)}, \
               {'id_min':id_min, 'id_max':id_max}
    


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

class RNA_Model(nn.Module):
    def __init__(self, dim=210, depth=24, head_size=7, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),  # Add a linear layer
            nn.ReLU(),  # Add an activation function if needed
            nn.Linear(dim, 2)  # Final projection layer
        )
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]
        
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        
        return x



def loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss


class GPT2Model(nn.Module):
    def __init__(self, dim=512, depth=48, head_size=8, num_layers=48):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                                       dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.transformer = nn.ModuleList(self.transformer_layers)
        self.proj_out = nn.Linear(dim, 2)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos

        for layer in self.transformer:
            x = layer(x, src_key_padding_mask=~mask)

        x = self.proj_out(x)

        return x


# class MAE(Metric):
#     def __init__(self): 
#         self.reset()
        
#     def reset(self): 
#         self.x,self.y = [],[]
        
#     def accumulate(self, learn):
#         x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
#         y = learn.y['react'][learn.y['mask']].clip(0,1)
#         self.x.append(x)
#         self.y.append(y)

#     @property
#     def value(self):
#         x,y = torch.cat(self.x,0),torch.cat(self.y,0)
#         loss = F.l1_loss(x, y, reduction='none')
#         loss = loss[~torch.isnan(loss)].mean()
#         return loss

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

class SimpleTFModel(pl.LightningModule):
    def __init__(self,learning_rate=8e-4):#,d_model, nhead, num_encoder_layers, num_decoder_layers,learning_rate=None):
        super(SimpleTFModel, self).__init__()
        self.transformer = RNA_Model()
        self.lr=learning_rate
        self.save_hyperparameters()
        
    def forward(self, src):
        output=self.transformer(src)
        return output
    def loss(self,pred,target):
        #print("pred",pred.shape)
        #print("target",target['react'].shape)
        p = pred[target['mask'][:,:pred.shape[1]]]
        y = target['react'][target['mask']].clip(0,1)
        loss = F.l1_loss(p, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss = self.loss(outputs,y)
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss =  self.loss(outputs,y)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        self.log("hp_metric",loss, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())

        # Assuming a threshold of 0.5 for binary classification
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        correct_predictions = (predicted_labels == y.squeeze()).float()
        accuracy = correct_predictions.mean()
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}
    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




if __name__ == "__main__":
    pl.seed_everything(42)
    task=sys.argv[1]
    if task=="train":

        

        d_model = 480
        nhead = 480
        num_encoder_layers = 124
        num_decoder_layers = 124
        learning_rate=1e-5


        lmodel=SimpleTFModel()
        
        datamodule = RNADataModule(batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_DATASET_FILE)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            filename='tf2-model-epoch_{epoch:02d}_val_loss_{val_loss:.2f}',
            every_n_epochs=1,
            dirpath=MODEL_DIR_PREFIX
        )
        # early_stop_callback = EarlyStopping(
        #     monitor='val_loss',
        #     patience=5,
        #     strict=False,
        #     verbose=False,
        #     mode='min'
        # )
        if sys.argv[2]=="start": 
        #validation_loss_callback = ValidationLossCallback()
            trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],
            accelerator=ACCELERATION, devices=DEVICES, 
            strategy="ddp")

            # Train the model limit_train_batches=0.1,limit_val_batches=200
            trainer.fit(lmodel, datamodule=datamodule)
        elif sys.argv[2]=="resume":
            file_name=sys.argv[3]
            SimpleTFModel.load_from_checkpoint(checkpoint_path=file_name)
            trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],
            accelerator=ACCELERATION, devices=DEVICES, strategy="ddp")

            # Train the model limit_train_batches=0.1,limit_val_batches=200
            trainer.fit(lmodel, datamodule=datamodule)

    
    elif  task=='test' :
        filter=False
        file_name=sys.argv[2]
        data=pd.read_csv(TEST_DATA)
        if filter==True:
            data=data[data['id_max'] > 267052521].reset_index(drop=True)
            outf=open("balance_submission.csv","w")
        print("Data length",len(data))
        device="cuda:1"
        test_dataset=RNA_TestDataset(data,device)
        test_dataloader = DataLoader(test_dataset, batch_size=PREDICT_BATCH_SIZE, shuffle=False)
        model=SimpleTFModel.load_from_checkpoint(file_name,map_location=torch.device('cuda:3'))
        # checkpoint = torch.load(file_name)
        # model.load_state_dict(checkpoint["state_dict"])
        # model.to(device)
        model.eval()
        if filter!=True:
            outf=open("submission.csv","w")
            header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"
            outf.write(header)
        for x,ids in tqdm(test_dataloader) :
            out=model(x)
            out=out.to('cpu')
            del x
            #torch.cuda.empty_cache()
            for index in range(0,out.size()[0]):
            #print(idmin.tolist())
            
                id_min=int(ids['id_min'][index])
                id_max=int(ids['id_max'][index])
                
                
                y=out[index].tolist()
                idx=0
                for i in range(id_min,id_max+1):

                    re_dms=round(y[idx][1],3)
                    if re_dms<=0:
                        re_dms=0.0
                    re_2a3=round(y[idx][0],3)
                    if re_2a3<=0:
                        re_2a3=0.0
                                
                    outf.write(str(i)+","+str(re_dms)+","+str(re_2a3)+"\n")
                    idx=idx+1
            
