from torch import nn

from src.models.causal_tft.tft_core import TFTCore

class TFTBaseline(TFTCore):
    def __init__(
        self,
        projection_length: int,
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
        kernel_size: int = 7
    ):
        """This class is an abstract class. It contains the key components of a TFT. It is also a pytorch lightning module, this 
        framework of pytorch used to quickly train the models. This class is used as a parent of every model we work on.

        Args:
            projection_length (int): Number of time steps to forecast at once
            horizon (int): Horizon of the prediction, how many days we want to predict at most.
            static_features_size (int): Number of features in the static metadata. This number should include the legnth of temporal_embedding_sizes
            temporal_features_size (int): Number of temporal features. This number should include the length of temporal_embedding_sizes. This number does not include 
            the trend size
            Temporal features are used both before and after the prediction time step
            target_size (int): Number of head at the end of the network. Defaults to 1.
            hidden_size (int): Size of the hidden state of the network. Defaults to 128.
            n_heads (int): Number of head for the attention mechanism. Defaults to 4.
            attn_dropout (float): Dropout rate for the attention mechanism. Defaults to 0.0.
            dropout (float): Dropout rate. Defaults to 0.1.
            learning_rate (float): initial lr. Defaults to 1e-3.
            static_embedding_sizes (list, optional): List of the maximum values for the categorical features in the static features. Defaults to None.
            temporal_embedding_sizes (list, optional): List of the maximum values for the categorical features in the future data. Defaults to None.
            trend_size (int): size of the trend (number of dimensions/channels). Defaults is 2
            n_static_layers (int): number of layers used to embed static features. Defaults to 2
            n_att_layers (int): number of attention layers

        """
        super().__init__()

        self.regression_loss = nn.MSELoss()
        self.projection_length = projection_length
        