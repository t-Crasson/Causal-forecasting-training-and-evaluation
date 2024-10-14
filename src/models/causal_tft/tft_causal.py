from typing import TypeVar, Generic

from torch.nn import functional as F

from src.models.causal_tft.tft_baseline import TFTBaseline
from torch import nn

from src.models.causal_tft.tft_core import TFTBackbone
from src.models.causal_tft.utils import create_sequential_layers
from src.models.causal_tft.treatment_encoder import AbstractTreatmentModule
from torch import Tensor
import torch

T = TypeVar("T", bound=AbstractTreatmentModule)

class CausalTFT(TFTBaseline, Generic[T]):

    def __init__(
        self,
        treatment_module_class: type[T],
        projection_length: int,
        last_nn: list[int],
        horizon: int,
        static_features_size: int,
        temporal_features_size: int,
        target_size: int = 1,
        hidden_size: int = 128,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        static_embedding_sizes: list | None = None,
        temporal_embedding_sizes: list | None = None,
        trend_size: int = 1,  # TODO rename in something like past_features_size
        n_static_layers: int = 2,
        n_att_layers: int = 4,
        conv_padding_size: int = 128,
        conv_blocks: int = 2,
        kernel_size: int = 7,
        treatment_max_value: int = 2,
        weight_decay: float = 1e-4,
    ):
        self.weight_decay = weight_decay
        super().__init__(
            projection_length=projection_length,
            last_nn=last_nn,
            horizon=horizon,
            static_features_size=static_features_size,
            temporal_features_size=temporal_features_size,
            target_size=target_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            learning_rate=learning_rate,
            static_embedding_sizes=static_embedding_sizes,
            temporal_embedding_sizes=temporal_embedding_sizes,
            trend_size=trend_size,
            n_static_layers=n_static_layers,
            n_att_layers=n_att_layers,
            conv_padding_size=conv_padding_size,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

        self.loss_regression = nn.MSELoss()
        

        self.m0_backbone = TFTBackbone(
            horizon=horizon,
            static_features_size=static_features_size,
            temporal_features_size=temporal_features_size,
            target_size=target_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            learning_rate=learning_rate,
            static_embedding_sizes=static_embedding_sizes,
            temporal_embedding_sizes=temporal_embedding_sizes,
            trend_size=trend_size,
            n_static_layers=n_static_layers,
            n_att_layers=n_att_layers,
            conv_padding_size=conv_padding_size,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

        e0_backbone = TFTBackbone(
            horizon=horizon,
            static_features_size=static_features_size,
            temporal_features_size=temporal_features_size,
            target_size=target_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            learning_rate=learning_rate,
            static_embedding_sizes=static_embedding_sizes,
            temporal_embedding_sizes=temporal_embedding_sizes,
            trend_size=trend_size,
            n_static_layers=n_static_layers,
            n_att_layers=n_att_layers,
            conv_padding_size=conv_padding_size,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

        theta_backbone = TFTBackbone(
            horizon=horizon,
            static_features_size=static_features_size,
            temporal_features_size=temporal_features_size,
            target_size=target_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            learning_rate=learning_rate,
            static_embedding_sizes=static_embedding_sizes,
            temporal_embedding_sizes=temporal_embedding_sizes,
            trend_size=trend_size,
            n_static_layers=n_static_layers,
            n_att_layers=n_att_layers,
            conv_padding_size=conv_padding_size,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

        self.m0_head = create_sequential_layers(last_nn, hidden_size, target_size)
        self.treatment_module: T = treatment_module_class(
            theta_backbone=theta_backbone,
            e0_backbone=e0_backbone,
            treatment_max_value=treatment_max_value,
            hidden_size=hidden_size,
            last_nn=last_nn
        )

        self.configure_optimizers()
        self.save_hyperparameters()

    @property
    def training_theta(self) -> bool:
        return self.treatment_module.training_theta

    @training_theta.setter
    def training_theta(self, value: bool):
        self.treatment_module.training_theta = value

    def train(self, mode: bool = True):
        super().train(mode)
        self.treatment_module.train(mode)

    def eval(self):
        super().eval()
        self.treatment_module.eval()

    def forward(self, windows_batch: dict[str, Tensor], tau: Tensor | None = None):

        if self.training and not self.training_theta:
            m0 = self.m0_head(self.m0_backbone.get_z(windows_batch, tau))
        else:
            with torch.no_grad():
                m0 = self.m0_head(self.m0_backbone.get_z(windows_batch, tau))
        e0, theta = self.treatment_module.forward(windows_batch, tau)

        return m0, e0, theta

    def format_batch_window(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        windows_batch, taus, y, active_entries, treatments = super().format_batch_window(batch)
        temporal = windows_batch["multivariate_exog"]
        for k in range(self.treatment_module.treatment_max):
            for i, tau in enumerate(taus):
                temporal[i, tau:, -1 - k] = -1
        windows_batch["multivariate_exog"] = temporal

        return windows_batch, taus, y, active_entries, treatments

    def orthogonal_forecast(self, y_shape: tuple[int, ...], m0: Tensor, e0: Tensor, theta: Tensor, treatments: Tensor):
        encoded_treatment = self.treatment_module.encode_treatments(y_shape, treatments, e0_compare=False)
        shift = torch.matmul((encoded_treatment - e0).unsqueeze(-2), theta.unsqueeze(-1)).squeeze(-1)
        y_orthogonal = m0 + shift
        return y_orthogonal

    def forecast(self, batch: dict[str, Tensor]):
        windows, taus, y, active_entries, treatments = self.format_batch_window(batch)
        m0, e0, theta = self.forward(windows, taus)
        return self.orthogonal_forecast(y.shape, m0, e0, theta, treatments)

    def loss(
        self, y: Tensor, treatment: Tensor, m0: Tensor, e0: Tensor, theta: Tensor, active_entries: Tensor
    ) -> tuple[Tensor, ...]:
        ##Process data to compute the losses
        loss_reg = self.loss_m0(y, m0, active_entries)
        loss_e0 = self.loss_e0(self.treatment_module.encode_treatments(y.shape, treatment, e0_compare=True), e0, active_entries)
        loss_orthogonal = self.loss_orthogonal(y, treatment, m0, e0, theta, active_entries)
        return loss_reg, loss_e0, loss_orthogonal

    def loss_m0(self, y: Tensor, m0: Tensor, active_entries: Tensor) -> Tensor:
        mse = F.mse_loss(y, m0, reduction="none")
        loss_regression = (mse * active_entries).sum() / active_entries.sum()
        return loss_regression

    def loss_e0(self, encoded_treatments: Tensor, e0: Tensor, active_entries: Tensor) -> Tensor:
        loss = self.treatment_module.loss_e0(encoded_treatments, e0)
        return (loss * active_entries.flatten()).sum() / active_entries.sum()

    def loss_orthogonal(
        self, y: Tensor, treatments: Tensor, m0: Tensor, e0: Tensor, theta: Tensor, active_entries: Tensor
    ) -> Tensor:
        y_orthogonal = self.orthogonal_forecast(y.shape, m0, e0, theta, treatments)
        mse = F.mse_loss(y, y_orthogonal, reduction="none")
        loss = (mse * active_entries).sum() / active_entries.sum()
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        windows_batch, taus, y, active_entries, treatments = super().format_batch_window(batch)
        m0, e0, theta = self.forward(windows_batch, taus)
        loss_reg, loss_e0, loss_orthogonal = self.loss(y, treatments, m0, e0, theta, active_entries)

        self.log("train_loss_reg_m_0", loss_reg, on_epoch=True, sync_dist=True)
        self.log("train_loss_e_0", loss_e0, on_epoch=True, sync_dist=True)
        self.log("train_orthogonal_loss", loss_orthogonal, on_epoch=True, sync_dist=True)

        if self.training_theta:
            loss = loss_orthogonal
        else:
            loss = loss_reg + loss_e0
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        windows_batch, taus, y, active_entries, treatments = super().format_batch_window(batch)
        m0, e0, theta = self.forward(windows_batch, taus)
        loss_reg, loss_e0, loss_orthogonal = self.loss(y, treatments, m0, e0, theta, active_entries)

        self.log("val_loss_reg_m_0", loss_reg, on_epoch=True, sync_dist=True)
        self.log("val_loss_e_0", loss_e0, on_epoch=True, sync_dist=True)
        self.log("val_orthogonal_loss", loss_orthogonal, on_epoch=True, sync_dist=True)
        # If we train m_0 and e_0 we compute the loss by taking their losses
        if self.training_theta:
            loss = loss_orthogonal
        else:
            loss = loss_reg + loss_e0
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Required by torch lightning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return self.optimizer




