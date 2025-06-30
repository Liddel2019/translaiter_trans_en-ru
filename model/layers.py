import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from utils.config import ConfigManager

# Configure logging for the layers module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Layer(nn.Module):
    """Base class for transformer layers with common functionality."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base layer with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.

        Raises:
            ValueError: If configuration initialization fails.

        Example:
            >>> config_manager = ConfigManager()
            >>> layer = Layer(config_manager.config)
        """
        super(Layer, self).__init__()
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Layer initialized with device: {self.device}")

    def initialize_weights(self) -> None:
        """
        Initialize layer weights using Xavier initialization.

        Raises:
            ValueError: If weight initialization fails.

        Example:
            >>> layer = Layer(config)
            >>> layer.initialize_weights()
        """
        try:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(module.weight)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            logger.info("Layer weights initialized with Xavier initialization")
        except Exception as e:
            logger.error(f"Failed to initialize weights: {str(e)}")
            raise ValueError(f"Failed to initialize weights: {str(e)}")

    def validate_input_shape(self, x: torch.Tensor, expected_dims: int = 3) -> bool:
        """
        Validate the shape of input tensor.

        Args:
            x (torch.Tensor): Input tensor to validate.
            expected_dims (int): Expected number of dimensions.

        Returns:
            bool: True if valid, False otherwise.

        Example:
            >>> layer = Layer(config)
            >>> valid = layer.validate_input_shape(torch.randn(2, 10, 512))
            >>> print(valid)
            True
        """
        try:
            if not isinstance(x, torch.Tensor):
                logger.error("Input must be a torch.Tensor")
                return False
            if x.dim() != expected_dims:
                logger.error(f"Input must be {expected_dims}-dimensional, got {x.dim()}")
                return False
            logger.debug(f"Validated input shape: {x.shape}")
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False


class MultiHeadAttentionLayer(Layer):
    """Implements multi-head attention mechanism with scaled dot-product attention."""

    def __init__(self, config: Dict[str, Any], hidden_size: int):
        """
        Initialize the multi-head attention layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            hidden_size (int): Hidden dimension size.

        Raises:
            ValueError: If configuration values are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> mha = MultiHeadAttentionLayer(config_manager.config, hidden_size=512)
        """
        super(MultiHeadAttentionLayer, self).__init__(config)
        try:
            self.num_heads = self.config_manager.get_config_value(
                "model.num_heads", config, default=8
            )
            logger.info(f"Loaded num_heads: {self.num_heads}")
            self.validate_config_value("num_heads", self.num_heads, int, positive=True)

            self.hidden_size = hidden_size
            logger.info(f"Loaded hidden_size: {self.hidden_size}")
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            logger.info(f"Loaded dropout_rate: {self.dropout_rate}")
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            if self.hidden_size % self.num_heads != 0:
                logger.error("hidden_size must be divisible by num_heads")
                raise ValueError("hidden_size must be divisible by num_heads")

            self.head_dim = self.hidden_size // self.num_heads
            self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.out_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.initialize_weights()
            self.to(self.device)
            logger.info(
                f"MultiHeadAttentionLayer initialized with {self.num_heads} heads, hidden size {self.hidden_size}")
        except Exception as e:
            logger.error(f"Failed to initialize MultiHeadAttentionLayer: {str(e)}")
            raise ValueError(f"Failed to initialize MultiHeadAttentionLayer: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                              positive: bool = False, range_bounds: Optional[Tuple[float, float]] = None) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string values are non-empty.
            positive (bool): If True, ensures numeric values are positive.
            range_bounds (Tuple[float, float]): Optional range for float values.

        Raises:
            ValueError: If validation fails.

        Example:
            >>> mha = MultiHeadAttentionLayer(config, 512)
            >>> mha.validate_config_value("num_heads", 8, int, positive=True)
        """
        try:
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, str) and not value.strip():
                logger.error(f"{key} cannot be empty")
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                logger.error(f"{key} must be positive")
                raise ValueError(f"{key} must be positive")
            if range_bounds and isinstance(value, float) and not (range_bounds[0] <= value <= range_bounds[1]):
                logger.error(f"{key} must be between {range_bounds[0]} and {range_bounds[1]}")
                raise ValueError(f"{key} must be between {range_bounds[0]} and {range_bounds[1]}")
            logger.debug(f"Validated {key}: {value}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def scale_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform scaled dot-product attention using PyTorch's optimized implementation.

        Args:
            q (torch.Tensor): Query tensor (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Key tensor (batch_size, num_heads, seq_len, head_dim).
            v (torch.Tensor): Value tensor (batch_size, num_heads, seq_len, head_dim).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and weights.

        Example:
            >>> mha = MultiHeadAttentionLayer(config, 512)
            >>> q, k, v = torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)
            >>> output, weights = mha.scale_dot_product_attention(q, k, v)
        """
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                attn_output = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=mask,
                    dropout_p=self.dropout_rate if self.training else 0.0
                )
                logger.debug(f"Computed scaled dot-product attention with output shape: {attn_output.shape}")
                return attn_output, None  # Weights not returned by optimized implementation
        except Exception as e:
            logger.error(f"Scaled dot-product attention failed: {str(e)}")
            raise ValueError(f"Scaled dot-product attention failed: {str(e)}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            q (torch.Tensor): Query tensor (batch_size, seq_len, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, seq_len, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, seq_len, hidden_size).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and weights.

        Example:
            >>> mha = MultiHeadAttentionLayer(config, 512)
            >>> x = torch.randn(2, 10, 512)
            >>> output, weights = mha(x, x, x)
            >>> print(output.shape)
            torch.Size([2, 10, 512])
        """
        try:
            if not self.validate_input_shape(q):
                raise ValueError("Invalid query input shape")
            if not self.validate_input_shape(k):
                raise ValueError("Invalid key input shape")
            if not self.validate_input_shape(v):
                raise ValueError("Invalid value input shape")

            batch_size, seq_len = q.size(0), q.size(1)
            q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_linear(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_linear(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_output, attn_weights = self.scale_dot_product_attention(q, k, v, mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            output = self.out_linear(attn_output)
            output = self.dropout(output)
            logger.debug(f"MultiHeadAttentionLayer forward output shape: {output.shape}")
            return output, attn_weights
        except Exception as e:
            logger.error(f"MultiHeadAttentionLayer forward failed: {str(e)}")
            raise ValueError(f"MultiHeadAttentionLayer forward failed: {str(e)}")

    def optimize_layer(self) -> None:
        """
        Optimize the layer using torch.compile for improved performance.

        Raises:
            ValueError: If optimization fails.

        Example:
            >>> mha = MultiHeadAttentionLayer(config, 512)
            >>> mha.optimize_layer()
        """
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
                logger.info("MultiHeadAttentionLayer optimized with torch.compile")
            else:
                logger.warning("torch.compile not available, skipping optimization")
        except Exception as e:
            logger.error(f"Failed to optimize layer: {str(e)}")
            raise ValueError(f"Failed to optimize layer: {str(e)}")


class FeedForwardLayer(Layer):
    """Implements a feed-forward neural network layer with GELU activation."""

    def __init__(self, config: Dict[str, Any], hidden_size: int):
        """
        Initialize the feed-forward layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            hidden_size (int): Hidden dimension size.

        Raises:
            ValueError: If configuration values are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> ffn = FeedForwardLayer(config_manager.config, hidden_size=512)
        """
        super(FeedForwardLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            logger.info(f"Loaded hidden_size: {self.hidden_size}")
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            logger.info(f"Loaded dropout_rate: {self.dropout_rate}")
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.linear1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
            self.linear2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.gelu = nn.GELU()
            self.initialize_weights()
            self.to(self.device)
            logger.info(f"FeedForwardLayer initialized with hidden size {self.hidden_size}")
        except Exception as e:
            logger.error(f"Failed to initialize FeedForwardLayer: {str(e)}")
            raise ValueError(f"Failed to initialize FeedForwardLayer: {str(e)}")

    def gelu_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after GELU activation.

        Example:
            >>> ffn = FeedForwardLayer(config, 512)
            >>> x = torch.randn(2, 10, 512)
            >>> output = ffn.gelu_activation(x)
        """
        try:
            output = self.gelu(x)
            logger.debug(f"GELU activation applied, output shape: {output.shape}")
            return output
        except Exception as e:
            logger.error(f"GELU activation failed: {str(e)}")
            raise ValueError(f"GELU activation failed: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor.

        Example:
            >>> ffn = FeedForwardLayer(config, 512)
            >>> x = torch.randn(2, 10, 512)
            >>> output = ffn(x)
            >>> print(output.shape)
            torch.Size([2, 10, 512])
        """
        try:
            if not self.validate_input_shape(x):
                raise ValueError("Invalid input shape")
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                x = self.linear1(x)
                x = self.gelu_activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.dropout(x)
                logger.debug(f"FeedForwardLayer forward output shape: {x.shape}")
                return x
        except Exception as e:
            logger.error(f"FeedForwardLayer forward failed: {str(e)}")
            raise ValueError(f"FeedForwardLayer forward failed: {str(e)}")


class EncoderLayer(Layer):
    """Implements a single encoder layer combining multi-head attention and feed-forward layers."""

    def __init__(self, config: Dict[str, Any], hidden_size: int):
        """
        Initialize the encoder layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            hidden_size (int): Hidden dimension size.

        Raises:
            ValueError: If configuration values are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> encoder = EncoderLayer(config_manager.config, hidden_size=512)
        """
        super(EncoderLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            logger.info(f"Loaded hidden_size: {self.hidden_size}")
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            logger.info(f"Loaded dropout_rate: {self.dropout_rate}")
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.self_attention = MultiHeadAttentionLayer(config, self.hidden_size)
            self.feed_forward = FeedForwardLayer(config, self.hidden_size)
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.initialize_weights()
            self.to(self.device)
            logger.info(f"EncoderLayer initialized with hidden size {self.hidden_size}")
        except Exception as e:
            logger.error(f"Failed to initialize EncoderLayer: {str(e)}")
            raise ValueError(f"Failed to initialize EncoderLayer: {str(e)}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, hidden_size).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output tensor.

        Example:
            >>> encoder = EncoderLayer(config, 512)
            >>> x = torch.randn(2, 10, 512)
            >>> output = encoder(x)
            >>> print(output.shape)
            torch.Size([2, 10, 512])
        """
        try:
            if not self.validate_input_shape(x):
                raise ValueError("Invalid input shape")
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                attn_output, _ = self.self_attention(x, x, x, mask)
                x = self.norm1(x + self.dropout(attn_output))
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
                logger.debug(f"EncoderLayer forward output shape: {x.shape}")
                return x
        except Exception as e:
            logger.error(f"EncoderLayer forward failed: {str(e)}")
            raise ValueError(f"EncoderLayer forward failed: {str(e)}")


class DecoderLayer(Layer):
    """Implements a single decoder layer combining masked self-attention, cross-attention, and feed-forward layers."""

    def __init__(self, config: Dict[str, Any], hidden_size: int):
        """
        Initialize the decoder layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            hidden_size (int): Hidden dimension size.

        Raises:
            ValueError: If configuration values are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> decoder = DecoderLayer(config_manager.config, hidden_size=512)
        """
        super(DecoderLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            logger.info(f"Loaded hidden_size: {self.hidden_size}")
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            logger.info(f"Loaded dropout_rate: {self.dropout_rate}")
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.self_attention = MultiHeadAttentionLayer(config, self.hidden_size)
            self.cross_attention = MultiHeadAttentionLayer(config, self.hidden_size)
            self.feed_forward = FeedForwardLayer(config, self.hidden_size)
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.norm3 = nn.LayerNorm(self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.initialize_weights()
            self.to(self.device)
            logger.info(f"DecoderLayer initialized with hidden size {self.hidden_size}")
        except Exception as e:
            logger.error(f"Failed to initialize DecoderLayer: {str(e)}")
            raise ValueError(f"Failed to initialize DecoderLayer: {str(e)}")

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the decoder layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, tgt_seq_len, hidden_size).
            memory (torch.Tensor): Encoder memory (batch_size, src_seq_len, hidden_size).
            tgt_mask (torch.Tensor, optional): Target self-attention mask.
            memory_mask (torch.Tensor, optional): Cross-attention mask.

        Returns:
            torch.Tensor: Output tensor.

        Example:
            >>> decoder = DecoderLayer(config, 512)
            >>> x = torch.randn(2, 10, 512)
            >>> memory = torch.randn(2, 10, 512)
            >>> output = decoder(x, memory)
            >>> print(output.shape)
            torch.Size([2, 10, 512])
        """
        try:
            if not self.validate_input_shape(x):
                raise ValueError("Invalid target input shape")
            if not self.validate_input_shape(memory):
                raise ValueError("Invalid memory input shape")
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                attn_output, _ = self.self_attention(x, x, x, tgt_mask)
                x = self.norm1(x + self.dropout(attn_output))
                attn_output, _ = self.cross_attention(x, memory, memory, memory_mask)
                x = self.norm2(x + self.dropout(attn_output))
                ff_output = self.feed_forward(x)
                x = self.norm3(x + self.dropout(ff_output))
                logger.debug(f"DecoderLayer forward output shape: {x.shape}")
                return x
        except Exception as e:
            logger.error(f"DecoderLayer forward failed: {str(e)}")
            raise ValueError(f"DecoderLayer forward failed: {str(e)}")

    def optimize_layer(self) -> None:
        """
        Optimize the decoder layer using torch.compile for improved performance.

        Raises:
            ValueError: If optimization fails.

        Example:
            >>> decoder = DecoderLayer(config, 512)
            >>> decoder.optimize_layer()
        """
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
                logger.info("DecoderLayer optimized with torch.compile")
            else:
                logger.warning("torch.compile not available, skipping optimization")
        except Exception as e:
            logger.error(f"Failed to optimize layer: {str(e)}")
            raise ValueError(f"Failed to optimize layer: {str(e)}")


if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        hidden_size = config_manager.get_config_value("model.hidden_size", default=512)

        # Test MultiHeadAttentionLayer
        mha = MultiHeadAttentionLayer(config, hidden_size)
        logger.info(
            f"MultiHeadAttentionLayer config: heads={mha.num_heads}, hidden_size={mha.hidden_size}, dropout={mha.dropout_rate}")
        x = torch.randn(2, 10, hidden_size).to(mha.device)
        output, _ = mha(x, x, x)
        logger.info(f"MultiHeadAttentionLayer output shape: {output.shape}")

        # Test FeedForwardLayer
        ffn = FeedForwardLayer(config, hidden_size)
        logger.info(f"FeedForwardLayer config: hidden_size={ffn.hidden_size}, dropout={ffn.dropout_rate}")
        output = ffn(x)
        logger.info(f"FeedForwardLayer output shape: {output.shape}")

        # Test EncoderLayer
        encoder = EncoderLayer(config, hidden_size)
        logger.info(f"EncoderLayer config: hidden_size={encoder.hidden_size}, dropout={encoder.dropout_rate}")
        output = encoder(x)
        logger.info(f"EncoderLayer output shape: {output.shape}")

        # Test DecoderLayer
        decoder = DecoderLayer(config, hidden_size)
        logger.info(f"DecoderLayer config: hidden_size={decoder.hidden_size}, dropout={decoder.dropout_rate}")
        output = decoder(x, x)
        logger.info(f"DecoderLayer output shape: {output.shape}")

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")