import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from models import *
import logging
from rnadatasets import *

logging.basicConfig(filename="predict_nn.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Prediction started")

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
# Instantiate the Lightning model
input_size = 1  # Each element in the sequence is a scalar
hidden_size = 128
output_size = 1

model2a3=SimpleNN.load_from_checkpoint("2a3_nn.ckpt",map_location=torch.device('cuda:0'))
model2a3.eval()

modeldms=SimpleNN.load_from_checkpoint("dms_nn.ckpt",map_location=torch.device('cuda:1'))
modeldms.eval()


data=pd.read_csv('../test_sequences.csv')


dataset=RNA_NNTestDataset(data)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

outf=open("submission.csv","w")
header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"

outf.write(header)
for x,ids in tqdm(test_dataloader) :
    #print(x.shape)
    with torch.no_grad():
        x_2=x.clone().detach()
        x=x.to("cuda:0")
        x_2=x_2.to("cuda:1")
        y_2a3 = model2a3(x).to("cpu").tolist()[0]
        y_dms = modeldms(x_2).to("cpu").tolist()[0]
        
        id_min=int(ids[0].tolist()[0])
        id_max=int(ids[1].tolist()[0])
        idx=0
        for i in range(id_min,id_max+1):
            try:
                re_dms=round(y_dms[idx],3)
            except:
                logging.info("A3:Not enough values for seq "+str(id_min))
                re_dms=0.0 
            if re_dms<=0:
                re_dms=0.0
            try:
                re_2a3=round(y_2a3[idx],3)
            except:
                logging.info("DMS:Not enough values for seq "+str(id_min))
                re_2a3=0.0

            if re_2a3<=0:
                re_2a3=0.0
                        
            line=str(i)+","+str(re_dms)+","+str(re_2a3)+"\n"
            outf.write(line)
            idx=idx+1
        
#outf.writelines(lines)