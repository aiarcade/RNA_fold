import sys
import torch.nn as nn
import torch.optim as optim
import torch
from settings import *
from models import *
from rnadatasets import *

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    pl.seed_everything(42, workers=True)
    experiment_type=sys.argv[1]
    print("Experiment type",experiment_type)

    if experiment_type=="2A3_MaP":
        datamodule=ProbDataModuleWithFixed500(target_dirs[1],batch_size=TRAIN_BATCH_SIZE)
        lmodel = BPPReactivityPredictorWithRNN()
    else:
        datamodule=ProbDataModuleWithFixed500(target_dirs[0],batch_size=TRAIN_BATCH_SIZE)
        lmodel = BPPReactivityPredictorWithRNN()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='bpps-model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX+experiment_type
    )

    print(type(lmodel))
    trainer = pl.Trainer(precision=16,max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],accelerator=ACCELERATION, devices=DEVICES, strategy="ddp",enable_progress_bar=True)

    # Train the model
    trainer.fit(lmodel, datamodule=datamodule)
