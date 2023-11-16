import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from models import *

from rnadatasets import *



# Instantiate the Lightning model
input_size = 1  # Each element in the sequence is a scalar
hidden_size = 128
output_size = 1

model2a3=SimpleRNN.load_from_checkpoint("2a3.ckpt",map_location=torch.device('cuda:0'),input_size=1, hidden_size=15,output_size=1,n_rnn_layers=132,linear_size=125,learning_rate=0.010777591568048049)
model2a3.eval()

modeldms=SimpleRNN.load_from_checkpoint("dms.ckpt",map_location=torch.device('cuda:1'),input_size=1, hidden_size=13,output_size=1,n_rnn_layers=95,linear_size=27,learning_rate=0.0040788051568258835)
modeldms.eval()


data=pd.read_csv('test_sequences.csv')


dataset=RNA_TestDataset(data)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

outf=open("submission.csv","w")
header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"

lines=[header]
for x,ids in tqdm(test_dataloader) :
    with torch.no_grad():
        x_2=x.clone().detach()
        x=x.to("cuda:0")
        x_2=x_2.to("cuda:1")
        y_2a3 = model2a3(x.unsqueeze(-1)).to("cpu").tolist()[0]
        y_dms = modeldms(x_2.unsqueeze(-1)).to("cpu").tolist()[0]
        
        id_min=int(ids[0].tolist()[0])
        id_max=int(ids[1].tolist()[0])
        idx=0
        for i in range(id_min,id_max+1):
            re_dms=round(y_dms[idx][0],3)
            if re_dms<=0:
                re_dms=0.0
            re_2a3=round(y_2a3[idx][0],3)
            if re_2a3<=0:
                re_2a3=0.0
                        
            lines.append(str(i)+","+str(re_dms)+","+str(re_2a3)+"\n")
            idx=idx+1
        
outf.writelines(lines)