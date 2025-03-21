from aba_flooding.model import Model
from aba_flooding.data import MyDataset


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
# find optimal learning rate
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt




def train():
    # https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
    data = pd.read_csv("data/raw/Regn2004-2025.csv", sep=";")
    data["Dato"] = pd.to_datetime(data["Dato"], format="%d.%m.%Y")
    
    # Create a more compact time index
    data = data.sort_values("Dato")  # Ensure data is sorted by date
    # Create a dense sequential index instead of using year*12+month
    data["time_idx"] = np.arange(len(data))
    
    data.dropna(subset=["Nedbor"], inplace=True)
    
    # Recalculate time_idx after dropping NaN values to ensure it remains consecutive
    data = data.reset_index(drop=True)
    data["time_idx"] = np.arange(len(data))
    
    # Rest of your code remains the same
    data["hour"] = data["Dato"].dt.hour.astype(str)
    data["weekday"] = data["Dato"].dt.weekday.astype(str)
    data["place_id"] = "Roskilde"

    max_encoder_length = 24
    
    training_cutoff = data["time_idx"].max() - 24 * 7  # keep last week for validation

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Nedbor",
        group_ids=["place_id"],
        min_encoder_length=max_encoder_length // 2,  # allow predictions without history
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=24 * 7,
        static_categoricals=["place_id"],
        time_varying_known_categoricals=["hour", "weekday"],
        target_normalizer=GroupNormalizer(
            groups=["place_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=7
    )
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=max(batch_size * 10, 2),  # Ensure at least 2 examples in a batch
        num_workers=7
    )

    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    MAE()(baseline_predictions.output, baseline_predictions.y)

    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator="cpu",
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # res = Tuner(trainer).lr_find(
    #     tft,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     max_lr=10.0,
    #     min_lr=1e-6,
    # )

    # print(f"suggested learning rate: {res.suggestion()}")
    # fig = res.plot(show=True, suggest=True)
    # fig.show()

    early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],
        logger=logger,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # save the model
    trainer.save_checkpoint("model_end.ckpt")

    # Evaluate the model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    print(MAE()(predictions.output, predictions.y))

    plt.plot(predictions.y[0].numpy(), label="actual")
    plt.plot(predictions.output[0].numpy(), label="forecast")
    plt.legend()
    plt.savefig("forecast.png")
    plt.close()

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(
        val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
    )
    # Wrap the plotting in a try-except block
    try:
        for idx in range(min(10, len(raw_predictions.x["encoder_target"]))):  # plot up to 10 examples
            fig = best_tft.plot_prediction(
                raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
            )
            fig.savefig(f"prediction_plot_{idx}.png")  # Save the plot as a PNG file
    except IndexError as e:
        print(f"Error in plotting individual predictions: {e}")
        print("Skipping individual prediction plots")

    # Similarly for the second set of plots
    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    try:
        mean_losses = SMAPE(reduction="none").loss(predictions.output, predictions.y[0]).mean(1)
        indices = mean_losses.argsort(descending=True)  # sort losses
        for idx in range(min(10, len(indices))):  # plot up to 10 examples
            fig = best_tft.plot_prediction(
                raw_predictions.x,
                raw_predictions.output,
                idx=indices[idx],
                add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles),
            )
            fig.savefig(f"prediction_plot_sorted_{idx}.png")
    except IndexError as e:
        print(f"Error in plotting sorted predictions: {e}")
        print("Skipping sorted prediction plots")


if __name__ == "__main__":
    train()
