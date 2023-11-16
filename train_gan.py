import sys
import torch.nn as nn
import torch.optim as optim
import torch
from settings import *
from models import *
from rnadatasets import *
from cgan import *


if __name__ == "__main__":

    pl.seed_everything(42)
    experiment_type=sys.argv[1]
    print("Experiment type",experiment_type)

    if experiment_type=="2A3_MaP":
       
        lmodel = CGAN()
    else:
       
        lmodel = CGAN()
    
    
    datamodule = ProbDataModule(target_dirs[0],batch_size=TRAIN_BATCH_SIZE)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='rnn-model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX+experiment_type
    )
    #validation_loss_callback = ValidationLossCallback()
    print(type(lmodel))
    trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS,accelerator=ACCELERATION, devices=DEVICES, strategy="ddp")

    # Train the model
    trainer.fit(lmodel, datamodule=datamodule)

