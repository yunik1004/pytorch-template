"""Train script
"""

import argparse
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader

# TODO: Change!!
from pathlib import Path
from lightning.pytorch.demos import Transformer


class TrainerModel(L.LightningModule):
    """Temporary model class for training and validation"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)

        # TODO: Change!!
        self.model = Transformer()
        if self.hparams.ema:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(self.hparams.ema_decay)
            )
            for param in self.ema_model.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # TODO: Change!!
        x, y = batch
        y_hat = self.model(x, y)
        loss = F.nll_loss(y_hat, y.view(-1))
        self.log_dict({"train/loss": loss})
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.hparams.ema:
            self.ema_model.update_parameters(self.model)

    def on_train_epoch_end(self):
        if self.hparams.ema:
            update_bn(self.trainer.train_dataloader, self.ema_model)

    def validation_step(self, batch, batch_idx):
        # TODO: Change!!
        x, y = batch
        y_hat = self.ema_model(x, y) if self.hparams.ema else self.model(x, y)
        loss = F.nll_loss(y_hat, y.view(-1))
        self.log_dict({"val/loss": loss}, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    seed_everything(args.seed)

    # Data
    # TODO: Change!!
    dataset = L.pytorch.demos.WikiText2(Path("./lightning_logs"))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False
    )

    # Model
    model = TrainerModel(args)

    # Logger
    logger = TensorBoardLogger(".")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.val_epoch_freq, save_top_k=-1
    )

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=FSDPStrategy(),
        callbacks=[checkpoint_callback],
        logger=False if args.validation else logger,
        deterministic=args.validation,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_epoch_freq,
    )

    if args.validation:
        trainer.validate(model, val_dataloader)
        return

    trainer.fit(model, train_dataloader, val_dataloader)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser(
        "Test script description",
    )
    parser.add_argument("--log_dir", type=str, default=".", help="tensorboard log dir")
    parser.add_argument(
        "--validation", action="store_true", help="perform validation only"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--ema", action="store_true", help="perform EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="maximum number of epochs"
    )
    parser.add_argument(
        "--val_epoch_freq", type=int, default=1, help="validation epoch frequency"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for data loading"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
