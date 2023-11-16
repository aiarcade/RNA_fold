from settings import *
from rnadatasets import *


dataset=StructureProbDataset(target_dirs[0])

for x,y in dataset:
    print(x)
    break
