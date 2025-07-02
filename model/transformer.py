import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from utils.config import ConfigManager
from utils.logger import Logger
from model.layers import MultiHeadAttentionLayer, FeedForwardLayer, EncoderLayer, DecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int):
        super(TransformerModel, self).__init__()
        self.config_manager = ConfigManager()
        self.logger = Logger(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.num_layers = self.config_manager.get_config_value(
                "model.num_layers", config, default=6
            )
            self.validate_config_value("num_layers", self.num_layers, int, positive=True)

            self.num_heads = self.config_manager.get_config_value(
                "model.num_heads", config, default=8
            )
            self.validate_config_value("num_heads", self.num_heads, int, positive=True)

            self.hidden_size = self.config_manager.get_config_value(
                "model.hidden_size", config, default=512
            )
            self.validate_config_value("hidden_size", self.hidden_size, int, positive=True)

            self.dropout_rate = self.config_manager.get_config_value(
                "model.dropout_rate", config, default=0.1
            )
            self.validate_config_value("dropout_rate", self.dropout_rate, float, range_bounds=(0, 1))

            self.checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "model.checkpoint_path", config, default="model/checkpoints"
                )
            )
            self.validate_config_value("checkpoint_path", self.checkpoint_path, str, non_empty=True)

            self.beam_size = self.config_manager.get_config_value(
                "training.beam_size", config, default=5
            )
            self.validate_config_value("beam_size", self.beam_size, int, positive=True)

            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", config, default=10
            )
            self.validate_config_value("max_length", self.max_length, int, positive=True)

        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize model configuration")
            raise ValueError(f"Model configuration initialization failed: {str(e)}")

        self.validate_config_value("src_vocab_size", src_vocab_size, int, positive=True)
        self.validate_config_value("tgt_vocab_size", tgt_vocab_size, int, positive=True)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        try:
            self.src_embedding = nn.Embedding(src_vocab_size, self.hidden_size)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.hidden_size)
            self.positional_encoding = self.create_positional_encoding()
            self.encoder = self.build_encoder()
            self.decoder = self.build_decoder()
            self.output_layer = nn.Linear(self.hidden_size, tgt_vocab_size)
            self.dropout = nn.Dropout(self.dropout_rate)

            self.validate_layer_config()
            self.initialize_layer_stack()
            self.to(self.device)
        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize model components")
            raise ValueError(f"Failed to initialize model components: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                              positive: bool = False, range_bounds: Optional[Tuple[float, float]] = None) -> None:
        try:
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {key}: expected {expected_type}")
            if non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{key} must be positive")
            if range_bounds and isinstance(value, float) and not (range_bounds[0] <= value <= range_bounds[1]):
                raise ValueError(f"{key} must be between {range_bounds[0]} and {range_bounds[1]}")
        except Exception as e:
            self.logger.log_exception(e, f"Validation failed for {key}")
            raise

    def validate_layer_config(self) -> None:
        try:
            if len(self.encoder) != self.num_layers:
                raise ValueError("Mismatch in encoder layer count")
            if len(self.decoder) != self.num_layers:
                raise ValueError("Mismatch in decoder layer count")
        except Exception as e:
            self.logger.log_exception(e, "Layer configuration validation failed")
            raise ValueError(f"Layer configuration validation failed: {str(e)}")

    def initialize_layer_stack(self) -> None:
        try:
            for layer in self.encoder:
                layer.initialize_weights()
            for layer in self.decoder:
                layer.initialize_weights()
            nn.init.xavier_uniform_(self.src_embedding.weight)
            nn.init.xavier_uniform_(self.tgt_embedding.weight)
            nn.init.xavier_uniform_(self.output_layer.weight)
        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize layer stack")
            raise ValueError(f"Failed to initialize layer stack: {str(e)}")

    def create_positional_encoding(self, max_len: int = 5000) -> torch.Tensor:
        try:
            pe = torch.zeros(max_len, self.hidden_size)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).to(self.device)
            return pe
        except Exception as e:
            self.logger.log_exception(e, "Failed to create positional encoding")
            raise ValueError(f"Failed to create positional encoding: {str(e)}")

    def build_encoder(self) -> nn.ModuleList:
        try:
            num_layers = self.config_manager.get_config_value("model.num_layers", default=6)
            self.validate_config_value("num_layers", num_layers, int, positive=True)
            encoder_layers = nn.ModuleList([
                EncoderLayer(self.config_manager.config, self.hidden_size) for _ in range(num_layers)
            ])
            return encoder_layers
        except Exception as e:
            self.logger.log_exception(e, "Failed to build encoder")
            raise ValueError(f"Failed to build encoder: {str(e)}")

    def build_decoder(self) -> nn.ModuleList:
        try:
            num_layers = self.config_manager.get_config_value("model.num_layers", default=6)
            self.validate_config_value("num_layers", num_layers, int, positive=True)
            decoder_layers = nn.ModuleList([
                DecoderLayer(self.config_manager.config, self.hidden_size) for _ in range(num_layers)
            ])
            return decoder_layers
        except Exception as e:
            self.logger.log_exception(e, "Failed to build decoder")
            raise ValueError(f"Failed to build decoder: {str(e)}")

    def validate_input_shape(self, input_ids: torch.Tensor, is_source: bool = True) -> bool:
        try:
            if not isinstance(input_ids, torch.Tensor):
                return False
            if input_ids.dim() != 2:
                return False
            if is_source:
                max_length = self.config_manager.get_config_value("tokenizer.max_length", default=10)
                if input_ids.size(-1) > max_length:
                    return False
            return True
        except Exception as e:
            self.logger.log_exception(e, "Input validation failed")
            return False

    def validate_embedding_output(self, emb: torch.Tensor) -> bool:
        try:
            if emb.dim() != 3:
                return False
            if emb.size(-1) != self.hidden_size:
                return False
            return True
        except Exception as e:
            self.logger.log_exception(e, "Embedding output validation failed")
            return False

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

            src_emb = self.src_embedding(src_ids) + self.positional_encoding[:, :src_ids.size(1), :]
            tgt_emb = self.tgt_embedding(tgt_ids) + self.positional_encoding[:, :tgt_ids.size(1), :]
            if not self.validate_embedding_output(src_emb):
                raise ValueError("Invalid source embedding output shape")
            if not self.validate_embedding_output(tgt_emb):
                raise ValueError("Invalid target embedding output shape")
            src_emb = self.dropout(src_emb)
            tgt_emb = self.dropout(tgt_emb)

            memory = src_emb
            for layer in self.encoder:
                memory = layer(memory, mask=src_mask)

            output = tgt_emb
            for layer in self.decoder:
                output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

            logits = self.output_layer(output)
            return logits
        except Exception as e:
            self.logger.log_exception(e, "Forward pass failed")
            raise ValueError(f"Forward pass failed: {str(e)}")

    def generate(self, src_ids: torch.Tensor, max_length: Optional[int] = None,
                 beam_size: Optional[int] = None) -> torch.Tensor:
        try:
            if not self.validate_input_shape(src_ids):
                raise ValueError("Invalid source input shape")

            max_length = max_length or self.config_manager.get_config_value("tokenizer.max_length", default=10)
            self.validate_config_value("max_length", max_length, int, positive=True)

            beam_size = beam_size or self.config_manager.get_config_value("training.beam_size", default=5)
            self.validate_config_value("beam_size", beam_size, int, positive=True)

            src_ids = src_ids.to(self.device)
            batch_size = src_ids.size(0)
            generated = torch.full((batch_size, 1), self.tgt_vocab_size - 1, dtype=torch.long,
                                   device=self.device)

            src_emb = self.src_embedding(src_ids) + self.positional_encoding[:, :src_ids.size(1), :]
            src_emb = self.dropout(src_emb)
            memory = src_emb
            for layer in self.encoder:
                memory = layer(memory)

            beams = [(generated, 0.0)]
            for _ in range(max_length - 1):
                new_beams = []
                for seq, score in beams:
                    if seq[:, -1].item() == self.tgt_vocab_size - 1:
                        new_beams.append((seq, score))
                        continue
                    tgt_emb = self.tgt_embedding(seq) + self.positional_encoding[:, :seq.size(1), :]
                    tgt_emb = self.dropout(tgt_emb)
                    output = tgt_emb
                    for layer in self.decoder:
                        output = layer(output, memory)
                    logits = self.output_layer(output[:, -1, :])
                    probs = F.softmax(logits, dim=-1)
                    top_probs, top_indices = probs.topk(beam_size, dim=-1)
                    for i in range(beam_size):
                        new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=-1)
                        new_score = score + torch.log(top_probs[:, i]).item()
                        new_beams.append((new_seq, new_score))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            return beams[0][0]
        except Exception as e:
            self.logger.log_exception(e, "Generation failed")
            raise ValueError(f"Generation failed: {str(e)}")

    def save_model(self, epoch: int = 0) -> None:
        try:
            checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value("model.checkpoint_path", default="model/checkpoints")
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_path, f"transformer_epoch_{epoch}.pth")
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
        except Exception as e:
            self.logger.log_exception(e, "Failed to save model")
            raise ValueError(f"Failed to save model: {str(e)}")

    def load_model(self, checkpoint_file: str) -> None:
        try:
            checkpoint_path = self.config_manager.get_absolute_path(checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.to(self.device)
        except Exception as e:
            self.logger.log_exception(e, "Failed to load model")
            raise ValueError(f"Failed to load model: {str(e)}")

    def get_model_summary(self) -> str:
        try:
            return str(self)
        except Exception as e:
            self.logger.log_exception(e, "Failed to generate model summary")
            return ""

    def get_parameter_count(self) -> int:
        try:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        except Exception as e:
            self.logger.log_exception(e, "Failed to count parameters")
            return 0

    def reset_weights(self) -> None:
        try:
            self.initialize_layer_stack()
        except Exception as e:
            self.logger.log_exception(e, "Failed to reset weights")
            raise ValueError(f"Failed to reset weights: {str(e)}")

    def optimize_model(self) -> None:
        try:
            if torch.__version__ >= "2.0":
                self.forward = torch.compile(self.forward)
                for layer in self.encoder:
                    if hasattr(layer, 'optimize_layer'):
                        layer.optimize_layer()
                for layer in self.decoder:
                    if hasattr(layer, 'optimize_layer'):
                        layer.optimize_layer()
            else:
                self.logger.log_message("WARNING", "torch.compile not available, skipping optimization")
        except Exception as e:
            self.logger.log_exception(e, "Failed to optimize model")
            raise ValueError(f"Failed to optimize model: {str(e)}")

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        logger = Logger(config)
        model = TransformerModel(config, src_vocab_size=32000, tgt_vocab_size=32000)
        src_ids = torch.zeros(2, 10, dtype=torch.long)
        tgt_ids = torch.zeros(2, 10, dtype=torch.long)
        output = model(src_ids, tgt_ids)
        generated = model.generate(src_ids)
        model.save_model(epoch=0)
        model.optimize_model()
    except Exception as e:
        logger.log_exception(e, "Test execution failed")