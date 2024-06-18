"""Train script
"""

import argparse
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader


class TrainerModel(L.LightningModule):
    """Temporary model class for training and validation"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)

        # TODO: Change!!
        from lightning.pytorch.demos import Transformer

        self.model = Transformer()
        if self.hparams.ema:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(self.hparams.ema_decay)
            )
            for param in self.ema_model.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # TODO: Change!!
        from torch.nn import functional as F

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
        from torch.nn import functional as F

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


class TrainerDataModule(L.LightningDataModule):
    """Temporary data module class for training and validation"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # TODO: Change!!
        from pathlib import Path

        self.dataset = L.pytorch.demos.WikiText2(Path("./lightning_logs"))

    def setup(self, stage: str):
        # TODO: Change!!
        if stage == "fit" or stage == "validate":
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            from torch.utils.data import random_split

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
        )


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    seed_everything(args.seed)

    dm = TrainerDataModule(args)
    model = TrainerModel(args)
    logger = TensorBoardLogger(args.log_dir)

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
        trainer.validate(model, datamodule=dm)
        return

    trainer.fit(model, datamodule=dm)


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
