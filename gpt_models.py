import functools
import warnings
from typing import Any, Optional, Tuple

import torch.optim
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning_utilities.core.overrides import is_overridden
import pytorch_lightning as pl

from nanogpt import * 


MINGPT_PRESETS = {
  
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}



class NanoGPT(pl.LightningModule):
    nanogpt: GPT

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        model_type: Optional[str] = "gpt2",
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        dropout: float = 0.1,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.build_nanogpt_configs()
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.betas=betas
        if not is_overridden("configure_sharded_model", self, pl.LightningModule):
            self.nanogpt = GPT(self.nanogpt_config)

    def build_nanogpt_configs(self) -> None:
        params = [
            self.hparams.n_layer,
            self.hparams.n_head,
            self.hparams.n_embd,
        ]

        params_given = all([el is not None for el in params])
        some_params_given = any([el is not None for el in params])

        if some_params_given and not params_given:
            raise ValueError(
                "Please provide all values for n_layer, n_head, and n_embd, or just model_type."
                f"Got n_layer={self.hparams.n_layer}, n_head={self.hparams.n_head}, "
                f"and n_embd={self.hparams.n_embd}."
            )

        if not params_given:
            # We take ownership of presets over minGPT here
            preset = MINGPT_PRESETS[self.hparams.model_type]
            self.hparams.update(preset)
            self.hparams.model_type = None

        self.nanogpt_config = GPTConfig()
        #self.merge_with_hparams(self.nanogpt_config)

        #self.nanogpt_trainer_config = mingpt.trainer.Trainer.get_default_config()
        #self.merge_with_hparams(self.nanogpt_trainer_config)

    # def merge_with_hparams(self, config: CfgNode) -> None:
    #     for k, v in self.hparams.items():
    #         if hasattr(config, k):
    #             setattr(config, k, v)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.nanogpt(idx, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.nanogpt.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas,
            device_type="cuda"

        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None
    ) -> torch.Tensor:
        return self.nanogpt.generate(idx, max_new_tokens, temperature, top_k)



