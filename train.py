import sys
import torch.nn as nn
import torch.optim as optim
import torch
from models import *
from rnadatasets import *



if __name__ == "__main__":

    pl.seed_everything(42)
    experiment_type=sys.argv[1]
    print("Experiment type",experiment_type)

    if experiment_type=="2A3_MaP":
        """
            Best trial:
            Value: 0.21786275506019592
            Params: 
                n_layers: 132
                h_units: 15
                l_units: 125
                learning_rate: 0.010777591568048049
        """
        lmodel = SimpleRNN(input_size=1, hidden_size=15,output_size=1,n_rnn_layers=132,linear_size=125,learning_rate=0.010777591568048049)
    else:
        """Best trial:
                Value: 0.22255480289459229
                Params: 
                    n_layers: 95
                    h_units: 13
                    l_units: 27
                    learning_rate: 0.0040788051568258835
        """
        lmodel = SimpleRNN(input_size=1, hidden_size=13,output_size=1,n_rnn_layers=95,linear_size=27,learning_rate=0.0040788051568258835)
    
    
    datamodule = RNADataModule(experiment=experiment_type, batch_size=128,data_file="train_data.parquet")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='rnn-model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        dirpath="./out"+experiment_type
    )
    #validation_loss_callback = ValidationLossCallback()
    print(type(lmodel))
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback],accelerator="gpu", devices=[0,1,2,3], strategy="ddp")

    # Train the model
    trainer.fit(lmodel, datamodule=datamodule)
