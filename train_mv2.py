import sys
import torch.nn as nn
import torch.optim as optim
import torch
from settings import *
from models import *
from rnadatasets import *


if __name__ == "__main__":

    pl.seed_everything(42, workers=True)
    experiment_type=sys.argv[1]
    print("Experiment type",experiment_type)

    if experiment_type=="2A3_MaP":
        datamodule=DataModuleRNAImageDataset(TRAIN_DATASET_FILE,'2A3_MaP',batch_size=TRAIN_BATCH_SIZE)
        lmodel = BPPReactivityPredictorWithMV2()
    else:
        datamodule=DataModuleRNAImageDataset(TRAIN_DATASET_FILE,'DMS_MaP',batch_size=TRAIN_BATCH_SIZE)
        lmodel = BPPReactivityPredictorWithMV2()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='mv2-model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX+experiment_type
    )

    print(type(lmodel))
    trainer = pl.Trainer(limit_train_batches=0.01,limit_val_batches=10 , max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],accelerator=ACCELERATION, devices=DEVICES, strategy="ddp",enable_progress_bar=True)

    # Train the model
    trainer.fit(lmodel, datamodule=datamodule)
