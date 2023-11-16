from settings import *
from rnadatasets import *


dataset=StructureProbDataset(SAMPLE_DATA,"2A3_MaP")

for x,y in dataset:
    print(x.shape)
