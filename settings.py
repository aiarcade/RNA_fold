import torch

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
ACCELERATION="gpu"
DEVICES=[0,1,2,3]
TRAIN_BATCH_SIZE=8
TRAIN_EPOCHS=100
PREDICT_BATCH_SIZE=4
TUNING_EPOCHS=5
TUNING_TRIALS=10
TUNING_BATCH_SIZE=128
MODEL_DIR_PREFIX="../out" 

TRAIN_DATASET_FILE="../train_data.parquet"
SAMPLE_DATA="../subset_data.parquet"
TEST_DATA="../test_sequences.csv"


experiments = ["DMS_MaP","2A3_MaP"]
target_dirs = ["../DMS_MaP","../2A3_MaP"] 