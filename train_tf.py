import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

from settings import *
import torch
import torch.nn as nn
from models import *
import pandas as pd
from rnadatasets import *
import sys


# Create a DataLoader
batch_size = TRAIN_BATCH_SIZE
num_samples = 1000
sequence_length = 480
input_size = 1

if __name__ == "__main__":

    pl.seed_everything(42)
    experiment_type=sys.argv[1]
    print("Experiment type",experiment_type)
    d_model = 480
    nhead = 120
    num_encoder_layers = 24
    num_decoder_layers = 24

    if experiment_type=="2A3_MaP":
        lmodel = SimpleTFModel(d_model, nhead, num_encoder_layers, num_decoder_layers)
    else:
        lmodel = SimpleTFModel(d_model, nhead, num_encoder_layers, num_decoder_layers)
    
    
    datamodule = RNATRDataModule(experiment=experiment_type, batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_DATASET_FILE)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='tf1-model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX+experiment_type
    ) 
    #validation_loss_callback = ValidationLossCallback()
    trainer = pl.Trainer(limit_train_batches=0.1,limit_val_batches=200,max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],accelerator=ACCELERATION, devices=DEVICES, strategy="ddp")

    # Train the model
    trainer.fit(lmodel, datamodule=datamodule)




# df=pd.read_parquet("../train_data.parquet")[1:2000]
# dataset = RNA_TRDataset(df,'2A3_MaP')
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# for x,y in dataloader:
#     print(x.shape)
#     print(y.shape)
#     break

# # Create the Transformer model


# model = 

# # Loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10

# for epoch in range(num_epochs):
#     total_loss = 0.0

#     for batch_X, batch_Y,l in dataloader:
#         # Zero the gradients
#         optimizer.zero_grad()

#         # Forward pass
#         output = model(batch_X, batch_Y)

#         # Compute the loss
#         loss = criterion(output, batch_Y)

#         # Backward pass
#         loss.backward()

#         # Update weights
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# # Small prediction example
# with torch.no_grad():
#     # Assuming you have a single sample for prediction
#     sample_X, sample_Y,l = next(iter(dataloader))


#     model.eval()    # Perform prediction
#     predicted_output = model(sample_X)

#     print("Sample X shape:", sample_X.shape)
#     print("Sample Y shape:", sample_Y.shape)
#     print("Predicted output shape:", predicted_output.shape)
#     print()
