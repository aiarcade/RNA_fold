import pandas as pd
import numpy as np
from arnie.bpps import bpps
import os
import multiprocessing as mp
from tqdm import tqdm

class Imageset():
    def __init__(self,file,experiment,target_dir):
        df=pd.read_parquet(file, engine='fastparquet')
        df=df[df['experiment_type'] == experiment]
        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        self.df=df.reset_index(drop=True)
        self.target_dir=target_dir
    def save(self):
        for index, row in tqdm(self.df.iterrows()):
            seq=self.encode_rna_sequence(row['sequence'])
            seq_len=len(row['sequence'])
            reactivity=np.array(row['reactivity'][0:seq_len])
            id=row['sequence_id']
            np.savez(self.target_dir+"/"+id+".npz",seq=seq,reactivity=reactivity)

    def encode_rna_sequence(self,sequence):
        return bpps(sequence, package='eternafold')

def createImages(experiment,src_filename,target_dir):
    print("starting conversion for",target_dir)
    dataset=Imageset(src_filename,experiment,target_dir)
    dataset.save()

        
if __name__ == "__main__":


    # Get the parent directory path
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Define the names of the directories to be checked/created
    experiments = ["DMS_MaP","2A3_MaP"]
    target_dirs = ["../DMS_MaP","../2A3_MaP"]
    src_filename ="../train_data.parquet"

    # Check and create directories if they do not exist
    for directory_name in target_dirs:
        directory_path = directory_name
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f'Directory "{directory_name}" created in {parent_directory}')
        else:
            print(f'Directory "{directory_name}" already exists in {parent_directory}')
    process1 = mp.Process(target=createImages, args=(experiments[0],src_filename,target_dirs[0]))
    process2 = mp.Process(target=createImages ,args=(experiments[1],src_filename,target_dirs[1]))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()