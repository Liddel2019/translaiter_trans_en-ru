import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from utils.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Layer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Layer, self).__init__()
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_weights(self) -> None:
        try:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(module.weight)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        except Exception as e:
            logger.error(f"Failed to initialize weights: {str(e)}")
            raise ValueError(f"Failed to initialize weights: {str(e)}")

    def validate_input_shape(self, x: torch.Tensor, expected_dims: int = 3) -> bool:
        try:
            if not isinstance(x, torch.Tensor):
                return False
            if x.dim() != expected_dims:
                return False
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                             positive: bool = False, range_bounds: Optional[Tuple[float, float]] = None) -> None:
        try:
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
            if non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{key} must be positive")
            if range_bounds and isinstance(value, float) and not (range_bounds[0] <= value <= range_bounds[1]):
                raise ValueError(f"{key} must be between {range_bounds[0]} and {range_bounds[1]}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

class MultiHeadAttentionLayer(Layer):
    def __init__(self, config: Dict[str, Any], hidden_size: int):
        super(MultiHeadAttentionLayer, self).__init__(config)
        try:
            self.num_heads = self.config_manager.get_config_value(
                "model.num_heads", config, default=8
            )
            self.validate_config_value("num_heads", self.num_heads, int, positive=True)

            self.hidden_size = hidden_size
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads")

            self.head_dim = self.hidden_size // self.num_heads
            self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.out_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.initialize_weights()
            self.to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize MultiHeadAttentionLayer: {str(e)}")
            raise ValueError(f"Failed to initialize MultiHeadAttentionLayer: {str(e)}")

    def scale_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                attn_output = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=mask,
                    dropout_p=self.dropout_rate if self.training else 0.0
                )
                return attn_output, None
        except Exception as e:
            logger.error(f"Scaled dot-product attention failed: {str(e)}")
            raise ValueError(f"Scaled dot-product attention failed: {str(e)}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
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
            return output, attn_weights
        except Exception as e:
            logger.error(f"MultiHeadAttentionLayer forward failed: {str(e)}")
            raise ValueError(f"MultiHeadAttentionLayer forward failed: {str(e)}")

    def optimize_layer(self) -> None:
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
            else:
                logger.warning("torch.compile not available, skipping optimization")
        except Exception as e:
            logger.error(f"Failed to optimize layer: {str(e)}")
            raise ValueError(f"Failed to optimize layer: {str(e)}")

class FeedForwardLayer(Layer):
    def __init__(self, config: Dict[str, Any], hidden_size: int):
        super(FeedForwardLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.linear1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
            self.linear2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.gelu = nn.GELU()
            self.initialize_weights()
            self.to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize FeedForwardLayer: {str(e)}")
            raise ValueError(f"Failed to initialize FeedForwardLayer: {str(e)}")

    def gelu_activation(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.gelu(x)
        except Exception as e:
            logger.error(f"GELU activation failed: {str(e)}")
            raise ValueError(f"GELU activation failed: {str(e)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if not self.validate_input_shape(x):
                raise ValueError("Invalid input shape")
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                x = self.linear1(x)
                x = self.gelu_activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.dropout(x)
                return x
        except Exception as e:
            logger.error(f"FeedForwardLayer forward failed: {str(e)}")
            raise ValueError(f"FeedForwardLayer forward failed: {str(e)}")

class EncoderLayer(Layer):
    def __init__(self, config: Dict[str, Any], hidden_size: int):
        super(EncoderLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.self_attention = MultiHeadAttentionLayer(config, self.hidden_size)
            self.feed_forward = FeedForwardLayer(config, self.hidden_size)
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.initialize_weights()
            self.to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize EncoderLayer: {str(e)}")
            raise ValueError(f"Failed to initialize EncoderLayer: {str(e)}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            if not self.validate_input_shape(x):
                raise ValueError("Invalid input shape")
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                attn_output, _ = self.self_attention(x, x, x, mask)
                x = self.norm1(x + self.dropout(attn_output))
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
                return x
        except Exception as e:
            logger.error(f"EncoderLayer forward failed: {str(e)}")
            raise ValueError(f"EncoderLayer forward failed: {str(e)}")

class DecoderLayer(Layer):
    def __init__(self, config: Dict[str, Any], hidden_size: int):
        super(DecoderLayer, self).__init__(config)
        try:
            self.hidden_size = hidden_size
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
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
        except Exception as e:
            logger.error(f"Failed to initialize DecoderLayer: {str(e)}")
            raise ValueError(f"Failed to initialize DecoderLayer: {str(e)}")

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
                return x
        except Exception as e:
            logger.error(f"DecoderLayer forward failed: {str(e)}")
            raise ValueError(f"DecoderLayer forward failed: {str(e)}")

    def optimize_layer(self) -> None:
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
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

        mha = MultiHeadAttentionLayer(config, hidden_size)
        x = torch.randn(2, 10, hidden_size).to(mha.device)
        output, _ = mha(x, x, x)

        ffn = FeedForwardLayer(config, hidden_size)
        output = ffn(x)

        encoder = EncoderLayer(config, hidden_size)
        output = encoder(x)

        decoder = DecoderLayer(config, hidden_size)
        output = decoder(x, x)
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")