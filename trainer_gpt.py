from settings import *
from rnadatasets import *
from gpt_models import *

pl.seed_everything(42)
extra_kwargs={"dropout" : 0.1}

model = NanoGPT(
    vocab_size=4,
    block_size=480,
    n_layer=12,
    n_head=12,
    n_embd=768,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    **extra_kwargs,
)

experiment_type='2A3_MaP'
datamodule = RNANNDataModule(experiment=experiment_type, batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_DATASET_FILE)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
    filename='nn-model-{epoch:02d}-{val_loss:.2f}',
    every_n_epochs=1,
    dirpath="../gptout"+experiment_type
) 
#validation_loss_callback = ValidationLossCallback()
trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],accelerator=ACCELERATION, devices=DEVICES, strategy="ddp")
trainer.fit(model, datamodule=datamodule)