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




model2a3=BPPReactivityPredictor.load_from_checkpoint("2a3.ckpt",map_location=torch.device('cuda:0'),input_channels=1, output_size=177 )
model2a3.eval()

modeldms=BPPReactivityPredictor.load_from_checkpoint("dms.ckpt",map_location=torch.device('cuda:1'),input_channels=1, output_size=177 )
modeldms.eval()


dataset=StructureProbTestDataset("../testdata/")
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

outf=open("../submission.csv","w")
header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"

lines=[header]
id_max=0
print("Starting prediction ...")
for x,ids,fid in test_dataloader :
    with torch.no_grad():
        x_2=x.clone().detach()
        x=x.to("cuda:0")
        x_2=x_2.to("cuda:1")
        y_2a3 = model2a3(x).to("cpu").tolist()[0]
        y_dms = modeldms(x_2).to("cpu").tolist()[0]
        id_min=int(ids[0][0])
        id_max=int(ids[0][1])
        idx=0
        for i in range(id_min,id_max+1):
            try:
                re_dms=round(y_dms[idx],3)
            except:
                print("A3:Not enough values for seq ",fid)
                re_dms=0.0 
            if re_dms<=0:
                re_dms=0.0
            try:
                re_2a3=round(y_2a3[idx],3)
            except:
                print("DMS:Not enough values for seq ",fid)
                re_2a3=0.0

            if re_2a3<=0:
                re_2a3=0.0
                        
            lines.append(str(i)+","+str(re_dms)+","+str(re_2a3)+"\n")
            idx=idx+1
    if fid%10000==0:
        print("Completed upto",fid)

outf.writelines(lines)