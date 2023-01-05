import pandas as pd 
import numpy as np
import torch
from typing import List, Tuple, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
import data_preprocessing as dp
from pathlib import Path
from functools import partial


# Constants
BATCH_SIZE: int = 128
WORKERS: int = 8
LR: float = 1e-3
EPOCHS: int = 50
QUANTILES: List[float] = [0.02, 0.25, 0.5, 0.75, 0.98]
SEED: int = 42


class BuoyDataset(torch.utils.data.Dataset):
    """ 
    torch Dataset for the Buoy Data 

    """

    def __init__(self, df: pd.DataFrame, target_label: str="AirTemperature"):
        df_c = df.copy()
        # we need to start at 0 for our embed layers
        self.station_ids = df_c.station_id.cat.codes.to_numpy(dtype=int)
        self.years = df_c.time.dt.year.to_numpy(dtype=int) - 2001
        self.days = df_c.time.dt.day_of_year.to_numpy(dtype=int) - 1
        self.hours = df_c.time.dt.hour.to_numpy(dtype=int) 
        df_c = df.drop(columns=["station_id", "time"])
        self.data = df_c.to_numpy(dtype=np.float32)
        self.targets = df_c[target_label].to_numpy(dtype=np.float32)
        
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        s_id = self.station_ids[idx]
        year = self.years[idx]
        doy = self.days[idx]
        hour = self.hours[idx]
        return s_id, year, doy, hour, self.data[idx], self.targets[idx]


def get_df_splits(data: pd.DataFrame, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # get train, test and validation sets
    df_, df_test = train_test_split(data, test_size=0.1, random_state=seed)
    df_train, df_val = train_test_split(df_, test_size=0.1, random_state=seed)
    return df_train, df_test, df_val


def get_dataloader(df: pd.DataFrame, shuffle, workers=WORKERS, batch_size=BATCH_SIZE) -> DataLoader:
    # put them into Datasets i do this by hand, because it is nice to have these objects for debugging purposes
    ds = BuoyDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return dl


class MyModel(pl.LightningModule):

    def __init__(self, 
        loss: callable, 
        lr: float, 
        architecture: torch.nn.Module,
        classification_head: torch.nn.Module,
        transformation_fn: callable = None
    ) -> None:
        super().__init__()
        # Embedding Layers are used to learn a representation for tabular data
        self.station_emb = torch.nn.Embedding(9, 9)
        self.year_emb = torch.nn.Embedding(23, 5) # apparently this is live data so year increases. 
        self.doy_emb = torch.nn.Embedding(366, 12) # leap years ;) 
        self.month_emb = torch.nn.Embedding(12, 6)
        self.hour_emb = torch.nn.Embedding(24, 12)
        self.lin_inp = torch.nn.Linear(11, 55)

        self.architecture = architecture
        self.classification_head = classification_head
        
        self.loss = loss
        self.lr = lr
        self.transformation_fn = transformation_fn # we need this for  
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_r2_score = torchmetrics.R2Score()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_r2_score = torchmetrics.R2Score()

    def forward(self, s_id, year, doy, hour, x) -> torch.Tensor:
        # calc_embeddings
        s_emb = self.station_emb(s_id)
        year_emb = self.year_emb(year)
        doy_emb = self.doy_emb(doy)
        hour_emb = self.hour_emb(hour)
        x_lin = self.lin_inp(x)
        # concat
        conv_in = torch.cat([s_emb, year_emb, doy_emb, hour_emb, x_lin], dim=1).unsqueeze(1)
        # let network do its job
        conv_out = self.architecture(conv_in)
        out = self.classification_head(conv_out)

        if self.transformation_fn is not None:
            out = self.transformation_fn(out)

        return out

    def _step(self, batch) -> torch.Tensor:
        s_id, year, doy, hour, x, y = batch
        pred = self.forward(s_id, year, doy, hour, x, ).squeeze()
        loss = self.loss(pred, y)
        return pred, loss

    def training_step(self, batch) -> torch.Tensor:
        pred, loss = self._step(batch)
        self.log("train/loss", loss)
        return loss
    
    def _eval_step(self, batch, mse, r2_score):
        pred, loss = self._step(batch)
        
        # TODO I use the transformation function as a way to indicate quantile regression, not nice. 
        if self.transformation_fn is not None:
            # Get the Point Prediction. In our case it is the middle 0.5 quantile. 
            pred = pred[::, 2]
        
        mse.update(pred, batch[-1])
        r2_score.update(pred, batch[-1])
        return loss
        
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._eval_step(batch, self.test_mse, self.test_r2_score)
        self.log("test", loss)

    def test_epoch_end(self, outputs) -> None:
        print(f"Test MSE: {self.test_mse.compute().data}")
        print(f"Test R2 Score: {self.test_r2_score.compute().data}")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._eval_step(batch, self.val_mse, self.val_r2_score)
        self.log("test", loss)

    def validation_epoch_end(self, outputs) -> None:
        print(f"Val MSE: {self.val_mse.compute().data}")
        print(f"Val R2 Score: {self.val_r2_score.compute().data}")
        
    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim


class QuantileLoss(torch.nn.Module):
    """
    Quantile Loss Function adapted from: 
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    Notes: 
        - If we just use the 0.5 Quantile it is equivalent to the MAE.
        - The middle Quantile is the Point Prediction. 
    """
    def __init__(self, quantiles) -> None:
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target) -> torch.Tensor:
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        loss.view
        return loss


def quantile_transformation(t: torch.Tensor, num_quantiles: int, output_channels: int) -> torch.Tensor:
    """
    Transforms the output into a shape the loss function can work with.
    Use it by partially applying it with functools.partial

    In a simple regression scenario this is not needed. But we need to reshape our output, in a scenario 
    where multiple parameters are predicted. 
    """
    return t.view(-1, num_quantiles, output_channels)


class SkipBlock(torch.nn.Module):
    """Skip Connection Block"""

    def __init__(self, in_channels, kernel_size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(self.in_channels, self.in_channels, kernel_size=kernel_size, padding="same"),
            torch.nn.BatchNorm1d(self.in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.in_channels, self.in_channels, kernel_size=kernel_size, padding="same"),
            torch.nn.BatchNorm1d(self.in_channels),
        )

    def forward(self, x):
        id = x
        out = self.block(x)
        out += id
        return torch.nn.functional.relu(out)


##########################################################################
# Architectures i want to test, with a corresponding classification head #
##########################################################################


# Resnet like architecture as an overkill solution
skip_architecture = torch.nn.Sequential(
    torch.nn.Conv1d(1, 10, kernel_size=5),
    torch.nn.BatchNorm1d(10),
    torch.nn.ReLU(),
    SkipBlock(10, 5),
    torch.nn.Conv1d(10, 15, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool1d(2),
    SkipBlock(15, 5),
    SkipBlock(15, 5),
    torch.nn.Conv1d(15, 20, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool1d(2),
    SkipBlock(20, 5),
    SkipBlock(20, 5),
    torch.nn.Conv1d(20, 30, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool1d(2),
    SkipBlock(30, 5),
    SkipBlock(30, 5),
    torch.nn.Dropout()
)


# Simple convolution architecture as a baseline and for quantile regression
simple_architecture = torch.nn.Sequential(
    torch.nn.Conv1d(1, 10, kernel_size=10), # 100 -> 95
    torch.nn.BatchNorm1d(10),
    torch.nn.ReLU(),
    torch.nn.Conv1d(10, 20, kernel_size=10),
    torch.nn.BatchNorm1d(20),
    torch.nn.ReLU(),
    torch.nn.Conv1d(20, 30, kernel_size=10),
    torch.nn.BatchNorm1d(30),
    torch.nn.ReLU(),
    torch.nn.Conv1d(30, 40, kernel_size=10),
    torch.nn.BatchNorm1d(40),
    torch.nn.ReLU(),
    torch.nn.Dropout()
)


########################
# Classification Heads #
########################


skip_head = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool1d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(30, 1),
)


classification_head = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool1d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(40, 1),
)


quantile_class_head = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(40, len(QUANTILES)),
    )


def train_model(model, trainer, dl_train, dl_val=None):
    # trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, log_every_n_steps=5, check_val_every_n_epoch=5)
    if dl_val is not None:
        trainer.fit(model, dl_train, val_dataloaders=[dl_val])
    else:
        trainer.fit(model, dl_train)


def test_model(model, trainer, dl_test):
    trainer.test(model, dataloaders=[dl_test])


def save_model(model, name, version):
    p = Path(f".{version}/saves/")
    p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / name)
    

#########################################
# Helper Functions to create the Models #
#########################################


def create_simple_model():
    loss = torch.nn.MSELoss()
    return MyModel(
        loss=loss, 
        lr=LR, 
        architecture=simple_architecture, 
        classification_head=classification_head,
        transformation_fn=None
    )


def create_skip_model():
    loss = torch.nn.MSELoss()
    return MyModel(
        loss=loss, 
        lr=LR, 
        architecture=skip_architecture, 
        classification_head=skip_head,
        transformation_fn=None
    )


def create_quantile_model(quantiles=QUANTILES) -> pl.LightningModule:
    tf = partial(quantile_transformation, num_quantiles=len(quantiles), output_channels=1)

    return MyModel(
        loss=QuantileLoss(quantiles=quantiles), 
        lr=LR, 
        architecture=simple_architecture, 
        classification_head=quantile_class_head,
        transformation_fn=tf
    )


def create_trainer() -> pl.Trainer:
    """Create a pytorch lightning trainer"""
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=EPOCHS, log_every_n_steps=5, check_val_every_n_epoch=5)
    else: 
        trainer = pl.Trainer(max_epochs=EPOCHS)
    return trainer


def main() -> None:
    """" Trains the models """
    data = dp.prepare_data()
    # We dont need the test set here
    df_train, _, df_val = get_df_splits(data, seed=SEED)
    dl_train = get_dataloader(df_train, shuffle=True)
    dl_val = get_dataloader(df_val, shuffle=False)

    # TODO automatic Versioning
    # Manual Versioning
    version = "version_0"

    mse_simple_model = create_simple_model()
    mse_skip_model = create_skip_model()
    quantile_model = create_quantile_model(QUANTILES)

    train_model(mse_simple_model, create_trainer(), dl_train, dl_val)
    train_model(mse_simple_model, create_trainer(), dl_train, dl_val)
    train_model(mse_simple_model, create_trainer(), dl_train, dl_val)

    save_model(mse_simple_model, "mse_simple_model", version)
    save_model(mse_skip_model, "mse_skip_model", version)
    save_model(quantile_model, "quantile_model", version)


if __name__ == "__main__":
    main()
