import argparse
import os
from typing import List
from typing import Optional


from packaging import version
import torch
from torch import nn
from torch import optim

from settings import *
import torch.nn.functional as F
import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from rnadatasets import *
from models import *
from settings import *






def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_rnnlayers = trial.suggest_int("n_layers",8,100)
    h_size   =  trial.suggest_int(f'h_units',8,64)
    l_size =  trial.suggest_int(f'l_units',8,64)
    lr = trial.suggest_float('learning_rate', 1e-5, 1000,log=True)
    model = SimpleRNN(input_size=1, hidden_size=h_size,output_size=1,n_rnn_layers=n_rnnlayers,linear_size=l_size,learning_rate=lr)
    datamodule = RNADataModule(experiment="DMS_MaP", batch_size=TUNING_BATCH_SIZE,data_file=SAMPLE_DATA)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=TUNING_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(hidden_size=h_size,n_rnn_layers=n_rnnlayers,linear_size=l_size,learning_rate=lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=TUNING_TRIALS, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

