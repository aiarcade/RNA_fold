import pandas as pd
import numpy as np
from arnie.bpps import bpps
import os
import multiprocessing as mp
from tqdm import tqdm

class Imageset():
    def __init__(self,df,target_dir):
        self.df=df
        self.target_dir=target_dir
    def save(self):
        for index, row in self.df.iterrows():
            seq=self.encode_rna_sequence(row['sequence'])
            seq_len=len(row['sequence'])
            reactivity=np.array(row['reactivity'][0:seq_len])
            id=row['sequence_id']
            np.savez(self.target_dir+"/"+id+".npz",seq=seq,reactivity=reactivity)
        

    def encode_rna_sequence(self,sequence):
        return bpps(sequence, package='eternafold')

def createImages(df,target_dir):
    dataset=Imageset(df,target_dir)
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
    
    srcdf=pd.read_parquet(src_filename, engine='fastparquet')
    processes=[]
    print("Total rows avilable in file",len(srcdf))
    for ex in range(len(experiments)): 
    
        df=srcdf[srcdf['experiment_type'] == experiments[ex]]
        df=df.drop_duplicates()
        df=df.drop(columns=df.filter(like='error').columns,axis=1)
        df=df.drop(['reads', 'SN_filter','signal_to_noise','experiment_type','dataset_name'],axis=1)
        reactivity_cols = df.filter(like='reactivity').columns
        df['reactivity'] = df[reactivity_cols].values.tolist()
        df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
        df=df.reset_index(drop=True)
        rows_per_df = len(df) // 16
        remaining_rows = len(df) % 16
        start_index = 0
        print("Total rows",experiments[ex],len(df))
        for i in range(16):
        # Calculate the end index for each split
            end_index = start_index + rows_per_df + (1 if i < remaining_rows else 0)
            part_df=df.iloc[start_index:end_index]
            print("Starting convertion from",start_index,end_index)
            process_c=mp.Process(target=createImages, args=(part_df,target_dirs[ex]))
            process_c.start()
            processes.append(process_c)
            start_index = end_index
    for p in processes:
        p.join()
