import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from models import *
from torchvision import transforms
from rnadatasets import *
import logging 

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(filename="predict.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Prediction started")

# model2a3=BPPReactivityPredictorWithMV2.load_from_checkpoint("2a3_v2.ckpt",map_location=torch.device('cuda:0') )
# model2a3.eval()

# # modeldms=BPPReactivityPredictorWithMV2.load_from_checkpoint("dms.ckpt",map_location=torch.device('cuda:1') )
# # modeldms.eval()


# dataset=StructureProbDatasetWithFixed500("../2A3_MaP")
# test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


# p_no=0
# for x,ids,file in test_dataloader :
#     with torch.no_grad():
#         x=x.to("cuda:0")
#         y_2a3 = model2a3(x).to("cpu").tolist()[0]
#         print(y_2a3)
#         print(file)
#         break
        
    #break
    
df=pd.read_parquet("../train_data.parquet")
dataset=RNA_NNDataset(df,"2A3_MaP")

model2a3= SimpleNN.load_from_checkpoint("2a3_nn.ckpt",map_location=torch.device('cuda:0') )
model2a3.eval()
for x,y in dataset:
    pred=model2a3(x.to("cuda:0"))
    errors = torch.abs(y - pred.to("cpu"))

# Calculate the mean error
    mean_error = errors.mean().item()

    print("Mean Error:", mean_error)


    

