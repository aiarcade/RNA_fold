import pandas as pd
from rnadatasets import *
from torch.utils.data import Dataset, DataLoader
# Read the CSV file into a DataFrame
# df = pd.read_csv('../submission.csv')

# # Identify duplicate values in the first column
# duplicate_rows = df[df.duplicated(subset=['id'], keep=False)]

# # Display the duplicate rows
# print("Duplicate Rows:")
# print(duplicate_rows)

dataset=StructureProbTestDataset("../testdata/")
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
mins=[]
maxs=[]

for x,ids,fid in test_dataloader:
    id_min=int(ids[0][0])
    id_max=int(ids[0][1])
    if id_min not in mins:
        mins.append(id_min)
    else:
        print("found duplicate",id_min,id_max,fid)
    if id_max not in maxs:
        maxs.append(id_max) 
    else:
        print("found duplicate",id_min,id_max,fid)