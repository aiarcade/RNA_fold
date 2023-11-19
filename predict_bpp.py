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

import logging 

logging.basicConfig(filename="predict.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Prediction started")

model2a3=BPPReactivityPredictorWithRNN.load_from_checkpoint("2a3.ckpt",map_location=torch.device('cuda:0') )
model2a3.eval()

modeldms=BPPReactivityPredictorWithRNN.load_from_checkpoint("dms.ckpt",map_location=torch.device('cuda:1') )
modeldms.eval()


dataset=StructureProbTestDataset500("../testdata/")
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

outf=open("../submission.csv","w")
header="id,reactivity_DMS_MaP,reactivity_2A3_MaP\n"

lines=[header]
id_max=0
print("Starting prediction ...")
p_no=0
for x,ids in test_dataloader :
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
                        
            lines.append(str(i)+","+str(re_dms)+","+str(re_2a3)+"\n")
            #print(lines)
            idx=idx+1
    if p_no%10000==0:
        print("Completed upto",p_no)
    p_no=p_no+1
    #break
    

outf.writelines(lines)