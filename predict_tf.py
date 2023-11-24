import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from settings import *
from tqdm import tqdm
from models import *


from rnadatasets import *


pl.seed_everything(42)
d_model = 480
nhead = 480
num_encoder_layers = 124
num_decoder_layers = 124



model2a3=SimpleTFModel(d_model, nhead, num_encoder_layers, num_decoder_layers)


modeldms=SimpleTFModel(d_model, nhead, num_encoder_layers, num_decoder_layers)

checkpoint2a3 = torch.load("2a3_tf.ckpt")
model2a3.load_state_dict(checkpoint2a3["state_dict"])

model2a3.to("cuda:0")

checkpointdms = torch.load("dms_tf.ckpt")
modeldms.load_state_dict(checkpointdms["state_dict"])
modeldms.to("cuda:1")

model2a3.eval()
modeldms.eval()
data=pd.read_csv('../test_sequences.csv')


dataset=RNA_TFTestDataset(data)
test_dataloader = DataLoader(dataset, batch_size=PREDICT_BATCH_SIZE, shuffle=False)

outf=open("submission.csv","w")
header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"
outf.write(header)
#lines=[header]
for x,idmin,idmax in tqdm(test_dataloader) :
    with torch.no_grad():
        x_2=x.clone().detach()
        x=x.to("cuda:0")
        x_2=x_2.to("cuda:1")
        y_2a3_b = model2a3(x).to("cpu")
        y_dms_b = modeldms(x_2).to("cpu")

        for index in range(0,y_2a3_b.size(0)):
            #print(idmin.tolist())
            
            id_min=int(idmin[index])
            id_max=int(idmax[index])
            
            y_dms=y_dms_b[index].tolist()
            y_2a3=y_2a3_b[index].tolist()
            idx=0
            for i in range(id_min,id_max+1):

                re_dms=round(y_dms[idx],3)
                if re_dms<=0:
                    re_dms=0.0
                re_2a3=round(y_2a3[idx],3)
                if re_2a3<=0:
                    re_2a3=0.0
                            
                outf.write(str(i)+","+str(re_dms)+","+str(re_2a3)+"\n")
                idx=idx+1
        
