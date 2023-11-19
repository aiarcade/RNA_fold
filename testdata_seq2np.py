import pandas as pd
import numpy as np
from arnie.bpps import bpps
import os
import multiprocessing as mp
from tqdm import tqdm

class Imageset():
    def __init__(self,df,target_dir,start_num):
        self.df=df
        self.target_dir=target_dir
        self.start_num=start_num
    def save(self):
        id=self.start_num
        for index, row in self.df.iterrows():
            ids=np.array([row['id_min'],row['id_max']])
            file_path=self.target_dir+"/"+str(ids[0])+".npz"
            if os.path.exists(file_path):
                id=id+1
                continue
            seq=self.encode_rna_sequence(row['sequence'])
            seq_len=len(row['sequence'])
                  
            np.savez(file_path,seq=seq,ids=ids)
            id=id+1
        

    def encode_rna_sequence(self,sequence):
        return bpps(sequence, package='eternafold')

def createImages(df,target_dir,start_num):
    dataset=Imageset(df,target_dir,start_num)
    dataset.save()
    print("Completed")

        
if __name__ == "__main__":


    # Get the parent directory path
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    target_dirs=["../testdata"]
    # Define the names of the directories to be checked/created
    src_filename ="../test_sequences.csv"

    # Check and create directories if they do not exist
    for directory_name in target_dirs:
        directory_path = directory_name
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f'Directory "{directory_name}" created in {parent_directory}')
        else:
            print(f'Directory "{directory_name}" already exists in {parent_directory}')
    
    srcdf=pd.read_csv(src_filename,)
    processes=[]
    print("Total rows avilable in file",len(srcdf))
    for ex in range(0,1): 
    
        df=srcdf
        df=df.drop_duplicates()
        rows_per_df = len(df) // 60
        remaining_rows = len(df) % 60
        start_index = 0
        print("Total rows",len(df))
        total=0
        for i in range(60):
        # Calculate the end index for each split
            end_index = start_index + rows_per_df + (1 if i < remaining_rows else 0)
            part_df=df.iloc[start_index:end_index]
            print("Starting convertion from",start_index,end_index)
            total=total+end_index-start_index
            process_c=mp.Process(target=createImages, args=(part_df,target_dirs[ex],start_index))
            process_c.start()
            processes.append(process_c)
            start_index = end_index
    for p in processes:
        p.join()
    print(total)