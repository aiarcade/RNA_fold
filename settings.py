import torch

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
ACCELERATION="gpu"
DEVICES=[0,1]
TRAIN_BATCH_SIZE=128
TRAIN_EPOCHS=10
TUNING_EPOCHS=3
TUNING_TRIALS=10
TUNING_BATCH_SIZE=128
MODEL_DIR_PREFIX="../out"

TRAIN_DATASET_FILE="../train_data.parquet"
SAMPLE_DATA="../subset_data.parquet"
TEST_DATA="../test_sequences.csv"
