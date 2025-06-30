import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from utils.config import ConfigManager
from model.layers import MultiHeadAttentionLayer, FeedForwardLayer, EncoderLayer, DecoderLayer

# Configure logging for the transformer module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    """Implements a transformer-based model for English-to-Russian translation in the translaiter_trans_en-ru project."""

    def __init__(self, config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int):
        """
        Initialize the transformer model with configuration settings using ConfigManager.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
            src_vocab_size (int): Vocabulary size for the source language (English).
            tgt_vocab_size (int): Vocabulary size for the target language (Russian).

        Raises:
            ValueError: If configuration values, vocabulary sizes, or layer initialization fails.

        Example:
            >>> config_manager = ConfigManager()
            >>> model = TransformerModel(config_manager.config, src_vocab_size=32000, tgt_vocab_size=32000)
            >>> print(model)
        """
        super(TransformerModel, self).__init__()
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Fetch configuration values using ConfigManager
        try:
            self.num_layers = self.config_manager.get_config_value(
                "model.num_layers", config, default=6
            )
            logger.info(f"Loaded num_layers: {self.num_layers}")
            self.validate_config_value("num_layers", self.num_layers, int, positive=True)

            self.num_heads = self.config_manager.get_config_value(
                "model.num_heads", config, default=8
            )
            logger.info(f"Loaded num_heads: {self.num_heads}")
            self.validate_config_value("num_heads", self.num_heads, int, positive=True)

            self.hidden_size = self.config_manager.get_config_value(
                "model.hidden_size", config, default=512
            )
            logger.info(f"Loaded hidden_size: {self.hidden_size}")
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            logger.info(f"Loaded dropout_rate: {self.dropout_rate}")
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "model.checkpoint_path", config, default="model/checkpoints"
                )
            )
            logger.info(f"Loaded checkpoint_path: {self.checkpoint_path}")
            self.validate_config_value("checkpoint_path", self.checkpoint_path, str, non_empty=True)

            self.beam_size = self.config_manager.get_config_value(
                "training.beam_size", config, default=5
            )
            logger.info(f"Loaded beam_size: {self.beam_size}")
            self.validate_config_value("beam_size", self.beam_size, int, positive=True)

            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", config, default=10
            )
            logger.info(f"Loaded max_length: {self.max_length}")
            self.validate_config_value("max_length", self.max_length, int, positive=True)

        except Exception as e:
            logger.error(f"Failed to initialize model configuration: {str(e)}")
            raise ValueError(f"Model configuration initialization failed: {str(e)}")

        # Validate vocabulary sizes
        self.validate_config_value("src_vocab_size", src_vocab_size, int, positive=True)
        self.validate_config_value("tgt_vocab_size", tgt_vocab_size, int, positive=True)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Initialize model components
        try:
            self.src_embedding = nn.Embedding(src_vocab_size, self.hidden_size)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.hidden_size)
            self.positional_encoding = self.create_positional_encoding()
            self.encoder = self.build_encoder()
            self.decoder = self.build_decoder()
            self.output_layer = nn.Linear(self.hidden_size, tgt_vocab_size)
            self.dropout = nn.Dropout(self.dropout_rate)

            # Validate layer stack
            self.validate_layer_config()
            self.initialize_layer_stack()
            self.to(self.device)
            logger.info(
                f"Transformer model initialized with {self.num_layers} layers, {self.num_heads} heads, hidden size {self.hidden_size}")
        except Exception as e:
            logger.error(f"Failed to initialize model components: {str(e)}")
            raise ValueError(f"Failed to initialize model components: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                              positive: bool = False, range_bounds: Optional[Tuple[float, float]] = None) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key for logging purposes.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string values are non-empty.
            positive (bool): If True, ensures numeric values are positive.
            range_bounds (Tuple[float, float]): Optional range for float values.

        Raises:
            ValueError: If the value does not meet validation criteria.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.validate_config_value("num_layers", 6, int, positive=True)
        """
        try:
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                raise ValueError(f"Invalid type for {key}: expected {expected_type}")
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

    def validate_layer_config(self) -> None:
        """
        Validate the configuration of encoder and decoder layer stacks.

        Raises:
            ValueError: If layer configuration is invalid or layer count mismatches.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.validate_layer_config()
        """
        try:
            if len(self.encoder) != self.num_layers:
                logger.error(f"Mismatch in encoder layer count: expected {self.num_layers}, got {len(self.encoder)}")
                raise ValueError("Mismatch in encoder layer count")
            if len(self.decoder) != self.num_layers:
                logger.error(f"Mismatch in decoder layer count: expected {self.num_layers}, got {len(self.decoder)}")
                raise ValueError("Mismatch in decoder layer count")
            logger.info("Layer configuration validated successfully")
        except Exception as e:
            logger.error(f"Layer configuration validation failed: {str(e)}")
            raise ValueError(f"Layer configuration validation failed: {str(e)}")

    def initialize_layer_stack(self) -> None:
        """
        Initialize weights for all layers in the encoder and decoder stacks.

        Raises:
            ValueError: If weight initialization fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.initialize_layer_stack()
        """
        try:
            for layer in self.encoder:
                layer.initialize_weights()
            for layer in self.decoder:
                layer.initialize_weights()
            nn.init.xavier_uniform_(self.src_embedding.weight)
            nn.init.xavier_uniform_(self.tgt_embedding.weight)
            nn.init.xavier_uniform_(self.output_layer.weight)
            logger.info("Initialized weights for encoder and decoder layer stacks")
        except Exception as e:
            logger.error(f"Failed to initialize layer stack: {str(e)}")
            raise ValueError(f"Failed to initialize layer stack: {str(e)}")

    def create_positional_encoding(self, max_len: int = 5000) -> torch.Tensor:
        """
        Create positional encodings for the transformer.

        Args:
            max_len (int): Maximum sequence length for positional encodings.

        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, max_len, hidden_size).

        Raises:
            ValueError: If positional encoding creation fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> pe = model.create_positional_encoding()
            >>> print(pe.shape)
            torch.Size([1, 5000, 512])
        """
        try:
            pe = torch.zeros(max_len, self.hidden_size)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).to(self.device)
            logger.debug(f"Created positional encoding of shape {pe.shape}")
            return pe
        except Exception as e:
            logger.error(f"Failed to create positional encoding: {str(e)}")
            raise ValueError(f"Failed to create positional encoding: {str(e)}")

    def build_encoder(self) -> nn.ModuleList:
        """
        Build the encoder layers of the transformer using configuration from ConfigManager and EncoderLayer from layers.py.

        Returns:
            nn.ModuleList: List of encoder layers.

        Raises:
            ValueError: If encoder building fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> encoder = model.build_encoder()
            >>> print(len(encoder))
            6
        """
        try:
            num_layers = self.config_manager.get_config_value("model.num_layers", default=6)
            logger.info(f"Building encoder with {num_layers} layers")
            self.validate_config_value("num_layers", num_layers, int, positive=True)
            encoder_layers = nn.ModuleList([
                EncoderLayer(self.config_manager.config, self.hidden_size) for _ in range(num_layers)
            ])
            logger.info(f"Built {num_layers} encoder layers using EncoderLayer")
            return encoder_layers
        except Exception as e:
            logger.error(f"Failed to build encoder: {str(e)}")
            raise ValueError(f"Failed to build encoder: {str(e)}")

    def build_decoder(self) -> nn.ModuleList:
        """
        Build the decoder layers of the transformer using configuration from ConfigManager and DecoderLayer from layers.py.

        Returns:
            nn.ModuleList: List of decoder layers.

        Raises:
            ValueError: If decoder building fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> decoder = model.build_decoder()
            >>> print(len(decoder))
            6
        """
        try:
            num_layers = self.config_manager.get_config_value("model.num_layers", default=6)
            logger.info(f"Building decoder with {num_layers} layers")
            self.validate_config_value("num_layers", num_layers, int, positive=True)
            decoder_layers = nn.ModuleList([
                DecoderLayer(self.config_manager.config, self.hidden_size) for _ in range(num_layers)
            ])
            logger.info(f"Built {num_layers} decoder layers using DecoderLayer")
            return decoder_layers
        except Exception as e:
            logger.error(f"Failed to build decoder: {str(e)}")
            raise ValueError(f"Failed to build decoder: {str(e)}")

    def validate_input_shape(self, input_ids: torch.Tensor, is_source: bool = True) -> bool:
        """
        Validate the shape of input tensor.

        Args:
            input_ids (torch.Tensor): Input tensor to validate.
            is_source (bool): Whether the input is source (True) or target (False).

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            ValueError: If input validation fails critically.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> valid = model.validate_input_shape(torch.zeros(2, 10, dtype=torch.long))
            >>> print(valid)
            True
        """
        try:
            if not isinstance(input_ids, torch.Tensor):
                logger.error("Input must be a torch.Tensor")
                return False
            if input_ids.dim() != 2:
                logger.error("Input must be 2-dimensional (batch_size, seq_length)")
                return False
            if is_source:
                max_length = self.config_manager.get_config_value("tokenizer.max_length", default=10)
                if input_ids.size(-1) > max_length:
                    logger.warning(f"Source input sequence length exceeds max_length: {max_length}")
                    return False
            logger.debug(f"Validated input shape: {input_ids.shape}")
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False

    def validate_embedding_output(self, emb: torch.Tensor) -> bool:
        """
        Validate the output shape of embedding layers.

        Args:
            emb (torch.Tensor): Embedding output tensor.

        Returns:
            bool: True if valid, False otherwise.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> emb = torch.randn(2, 10, 512)
            >>> valid = model.validate_embedding_output(emb)
            >>> print(valid)
            True
        """
        try:
            if emb.dim() != 3:
                logger.error(f"Embedding output must be 3-dimensional, got {emb.dim()}")
                return False
            if emb.size(-1) != self.hidden_size:
                logger.error(
                    f"Embedding output dimension must match hidden_size: {self.hidden_size}, got {emb.size(-1)}")
                return False
            logger.debug(f"Validated embedding output shape: {emb.shape}")
            return True
        except Exception as e:
            logger.error(f"Embedding output validation failed: {str(e)}")
            return False

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer model using layers from layers.py.

        Args:
            src_ids (torch.Tensor): Source input token IDs (batch_size, src_seq_len).
            tgt_ids (torch.Tensor): Target input token IDs (batch_size, tgt_seq_len).
            src_mask (torch.Tensor, optional): Source padding mask.
            tgt_mask (torch.Tensor, optional): Target attention mask (causal).

        Returns:
            torch.Tensor: Output logits (batch_size, tgt_seq_len, tgt_vocab_size).

        Raises:
            ValueError: If input shapes are invalid or model components are not initialized.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> output = model.forward(torch.zeros(2, 10, dtype=torch.long), torch.zeros(2, 10, dtype=torch.long))
            >>> print(output.shape)
            torch.Size([2, 10, 32000])
        """
        try:
            if not self.validate_input_shape(src_ids):
                raise ValueError("Invalid source input shape")
            if not self.validate_input_shape(tgt_ids, is_source=False):
                raise ValueError("Invalid target input shape")

            src_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(self.device)

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Embeddings and positional encoding
                src_emb = self.src_embedding(src_ids) + self.positional_encoding[:, :src_ids.size(1), :]
                tgt_emb = self.tgt_embedding(tgt_ids) + self.positional_encoding[:, :tgt_ids.size(1), :]
                if not self.validate_embedding_output(src_emb):
                    raise ValueError("Invalid source embedding output shape")
                if not self.validate_embedding_output(tgt_emb):
                    raise ValueError("Invalid target embedding output shape")
                src_emb = self.dropout(src_emb)
                tgt_emb = self.dropout(tgt_emb)
                logger.debug("Applied embeddings and positional encodings")

                # Encoder
                memory = src_emb
                for i, layer in enumerate(self.encoder):
                    memory = layer(memory, mask=src_mask)
                    logger.debug(f"Processed encoder layer {i + 1}/{self.num_layers}, output shape: {memory.shape}")
                logger.debug("Completed encoder processing")

                # Decoder
                output = tgt_emb
                for i, layer in enumerate(self.decoder):
                    output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
                    logger.debug(f"Processed decoder layer {i + 1}/{self.num_layers}, output shape: {output.shape}")
                logger.debug("Completed decoder processing")

                # Output layer
                logits = self.output_layer(output)
                logger.debug(f"Forward pass completed, output shape: {logits.shape}")
                return logits
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ValueError(f"Forward pass failed: {str(e)}")

    def generate(self, src_ids: torch.Tensor, max_length: Optional[int] = None,
                 beam_size: Optional[int] = None) -> torch.Tensor:
        """
        Generate translations using beam search with configuration from ConfigManager.

        Args:
            src_ids (torch.Tensor): Source input token IDs (batch_size, src_seq_len).
            max_length (int, optional): Maximum length for generated sequence. Defaults to config max_length.
            beam_size (int, optional): Beam size for search. Defaults to config beam_size.

        Returns:
            torch.Tensor: Generated token IDs (batch_size, generated_seq_len).

        Raises:
            ValueError: If input shape is invalid or generation fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> generated = model.generate(torch.zeros(1, 10, dtype=torch.long))
            >>> print(generated.shape)
            torch.Size([1, 10])
        """
        try:
            if not self.validate_input_shape(src_ids):
                raise ValueError("Invalid source input shape")

            max_length = max_length or self.config_manager.get_config_value("tokenizer.max_length", default=10)
            logger.info(f"Using max_length for generation: {max_length}")
            self.validate_config_value("max_length", max_length, int, positive=True)

            beam_size = beam_size or self.config_manager.get_config_value("training.beam_size", default=5)
            logger.info(f"Using beam_size for generation: {beam_size}")
            self.validate_config_value("beam_size", beam_size, int, positive=True)

            src_ids = src_ids.to(self.device)
            batch_size = src_ids.size(0)
            generated = torch.full((batch_size, 1), self.tgt_vocab_size - 1, dtype=torch.long,
                                   device=self.device)  # Start token

            # Encoder pass
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                src_emb = self.src_embedding(src_ids) + self.positional_encoding[:, :src_ids.size(1), :]
                src_emb = self.dropout(src_emb)
                memory = src_emb
                for i, layer in enumerate(self.encoder):
                    memory = layer(memory)
                    logger.debug(f"Encoder layer {i + 1} output shape: {memory.shape}")

                # Beam search
                beams = [(generated, 0.0)]  # (sequence, score)
                for _ in range(max_length - 1):
                    new_beams = []
                    for seq, score in beams:
                        if seq[:, -1].item() == self.tgt_vocab_size - 1:  # End token
                            new_beams.append((seq, score))
                            continue
                        tgt_emb = self.tgt_embedding(seq) + self.positional_encoding[:, :seq.size(1), :]
                        tgt_emb = self.dropout(tgt_emb)
                        output = tgt_emb
                        for i, layer in enumerate(self.decoder):
                            output = layer(output, memory)
                            logger.debug(f"Decoder layer {i + 1} output shape: {output.shape}")
                        logits = self.output_layer(output[:, -1, :])
                        probs = F.softmax(logits, dim=-1)
                        top_probs, top_indices = probs.topk(beam_size, dim=-1)
                        for i in range(beam_size):
                            new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=-1)
                            new_score = score + torch.log(top_probs[:, i]).item()
                            new_beams.append((new_seq, new_score))
                    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

                best_seq = beams[0][0]
                logger.info(f"Generated sequence with shape: {best_seq.shape}")
                return best_seq
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise ValueError(f"Generation failed: {str(e)}")

    def save_model(self, epoch: int = 0) -> None:
        """
        Save the model state to the checkpoint path using ConfigManager for path resolution.

        Args:
            epoch (int): Current epoch number for checkpoint naming.

        Raises:
            ValueError: If saving the model fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.save_model(epoch=1)
        """
        try:
            checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value("model.checkpoint_path", default="model/checkpoints")
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_path, f"transformer_epoch_{epoch}.pth")
            logger.info(f"Saving model to {checkpoint_file}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'config': {
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'hidden_size': self.hidden_size,
                    'dropout_rate': self.dropout_rate,
                    'src_vocab_size': self.src_vocab_size,
                    'tgt_vocab_size': self.tgt_vocab_size
                }
            }, checkpoint_file)
            logger.info(f"Model successfully saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ValueError(f"Failed to save model: {str(e)}")

    def load_model(self, checkpoint_file: str) -> None:
        """
        Load the model state from a checkpoint file using ConfigManager for path resolution.

        Args:
            checkpoint_file (str): Path to the checkpoint file.

        Raises:
            ValueError: If loading the model fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.load_model("model/checkpoints/transformer_epoch_1.pth")
        """
        try:
            checkpoint_path = self.config_manager.get_absolute_path(checkpoint_file)
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.to(self.device)
            logger.info(f"Model successfully loaded from {checkpoint_path}, epoch {checkpoint['epoch']}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def get_model_summary(self) -> str:
        """
        Generate a summary of the model architecture, including layer details.

        Returns:
            str: String representation of the model architecture.

        Raises:
            ValueError: If summary generation fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> print(model.get_model_summary())
            TransformerModel(
                (src_embedding): Embedding(32000, 512)
                ...
            )
        """
        try:
            summary = str(self)
            logger.debug("Generated model summary")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate model summary: {str(e)}")
            return ""

    def get_parameter_count(self) -> int:
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> param_count = model.get_parameter_count()
            >>> print(param_count)
            12345678
        """
        try:
            param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.debug(f"Total trainable parameters: {param_count}")
            return param_count
        except Exception as e:
            logger.error(f"Failed to count parameters: {str(e)}")
            return 0

    def reset_weights(self) -> None:
        """
        Reset model weights to their initial state.

        Raises:
            ValueError: If weight reset fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.reset_weights()
        """
        try:
            self.initialize_layer_stack()
            logger.info("Model weights reset to initial state")
        except Exception as e:
            logger.error(f"Failed to reset weights: {str(e)}")
            raise ValueError(f"Failed to reset weights: {str(e)}")

    def optimize_model(self) -> None:
        """
        Optimize the model using torch.compile for improved performance.

        Raises:
            ValueError: If optimization fails.

        Example:
            >>> model = TransformerModel(config, 32000, 32000)
            >>> model.optimize_model()
        """
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
                for layer in self.encoder:
                    if hasattr(layer, 'optimize_layer'):
                        layer.optimize_layer()
                for layer in self.decoder:
                    if hasattr(layer, 'optimize_layer'):
                        layer.optimize_layer()
                logger.info("Model optimized with torch.compile")
            else:
                logger.warning("torch.compile not available, skipping optimization")
        except Exception as e:
            logger.error(f"Failed to optimize model: {str(e)}")
            raise ValueError(f"Failed to optimize model: {str(e)}")


if __name__ == "__main__":
    # Example usage for testing
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        model = TransformerModel(config, src_vocab_size=32000, tgt_vocab_size=32000)
        logger.info("Model configuration:")
        logger.info(f"Number of layers: {model.num_layers}")
        logger.info(f"Number of heads: {model.num_heads}")
        logger.info(f"Hidden size: {model.hidden_size}")
        logger.info(f"Dropout rate: {model.dropout_rate}")
        logger.info(f"Checkpoint path: {model.checkpoint_path}")
        logger.info(f"Beam size: {model.beam_size}")
        logger.info(f"Max length: {model.max_length}")
        logger.info(f"Encoder layer count: {len(model.encoder)}")
        logger.info(f"Decoder layer count: {len(model.decoder)}")
        logger.info(f"Model summary:\n{model.get_model_summary()}")
        logger.info(f"Total parameters: {model.get_parameter_count()}")
        if model.validate_model_config():
            logger.info("Model configuration validated successfully")

        # Test forward pass
        src_ids = torch.zeros(2, 10, dtype=torch.long)
        tgt_ids = torch.zeros(2, 10, dtype=torch.long)
        output = model(src_ids, tgt_ids)
        logger.info(f"Forward pass output shape: {output.shape}")

        # Test generation
        generated = model.generate(src_ids)
        logger.info(f"Generated sequence shape: {generated.shape}")

        # Test saving
        model.save_model(epoch=0)

        # Test optimization
        model.optimize_model()
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")