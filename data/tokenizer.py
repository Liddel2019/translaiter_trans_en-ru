import os
import logging
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
from transformers import AutoTokenizer
from utils.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationTokenizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_manager = ConfigManager()
        self.tokenizer = None
        self.is_cached = False

        try:
            self.tokenizer_name = self.config_manager.get_config_value(
                "tokenizer.tokenizer_name", self.config, default="Helsinki-NLP/opus-mt-en-ru"
            )
            self.validate_config_value("tokenizer_name", self.tokenizer_name, str, non_empty=True)

            self.tokenizer_cache_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "tokenizer.tokenizer_cache_path", self.config, default="data/tokenizer_cache"
                )
            )
            self.validate_config_value("tokenizer_cache_path", self.tokenizer_cache_path, str, non_empty=True)

            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", self.config, default=10
            )
            self.validate_config_value("max_length", self.max_length, int, positive=True)

            self.vocab_size = self.config_manager.get_config_value(
                "tokenizer.vocab_size", self.config, default=32000
            )
            self.validate_config_value("vocab_size", self.vocab_size, int, positive=True)

            self._initialize()
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer configuration: {str(e)}")
            raise ValueError(f"Tokenizer initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False, positive: bool = False) -> None:
        try:
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {key}: expected {expected_type}")
            if non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{key} must be positive")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def _initialize(self) -> None:
        try:
            self.tokenizer_name = self.config_manager.get_config_value(
                "tokenizer.tokenizer_name", self.config, default="Helsinki-NLP/opus-mt-en-ru"
            )
            self.tokenizer_cache_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "tokenizer.tokenizer_cache_path", self.config, default="data/tokenizer_cache"
                )
            )
            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", self.config, default=10
            )
            self.vocab_size = self.config_manager.get_config_value(
                "tokenizer.vocab_size", self.config, default=32000
            )

            if self.load_cache():
                self.is_cached = True
            else:
                self.load_tokenizer()
                self.save_cache()
            if not self.validate_tokenizer():
                raise ValueError("Invalid tokenizer configuration")
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

    def load_tokenizer(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=self.tokenizer_cache_path,
                max_length=self.max_length,
                use_fast=True
            )
            actual_vocab_size = len(self.tokenizer)
            if actual_vocab_size != self.vocab_size:
                self.vocab_size = actual_vocab_size
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_name}: {str(e)}")
            raise ValueError(f"Failed to load tokenizer: {str(e)}")

    def encode(self, text: str) -> torch.Tensor:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not isinstance(text, str) or not text.strip():
                return torch.tensor([], dtype=torch.long)

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return encoding["input_ids"].squeeze(0)
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise ValueError(f"Failed to encode text: {str(e)}")

    def decode(self, token_ids: torch.Tensor) -> str:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
                return ""
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()

            decoded_text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return decoded_text.strip()
        except Exception as e:
            logger.error(f"Failed to decode token IDs: {str(e)}")
            raise ValueError(f"Failed to decode token IDs: {str(e)}")

    def save_cache(self) -> None:
        try:
            os.makedirs(self.tokenizer_cache_path, exist_ok=True)
            cache_file = os.path.join(self.tokenizer_cache_path, f"{self.tokenizer_name.replace('/', '_')}_cache.json")
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'tokenizer_name': self.tokenizer_name,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            self.tokenizer.save_pretrained(self.tokenizer_cache_path)
        except Exception as e:
            logger.error(f"Failed to save tokenizer cache: {str(e)}")
            raise ValueError(f"Failed to save tokenizer cache: {str(e)}")

    def load_cache(self) -> bool:
        try:
            cache_file = os.path.join(self.tokenizer_cache_path, f"{self.tokenizer_name.replace('/', '_')}_cache.json")
            if not os.path.exists(cache_file):
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if cache_data.get('tokenizer_name') != self.tokenizer_name:
                return False

            cache_time = datetime.fromisoformat(cache_data.get('timestamp'))
            if (datetime.now() - cache_time).days > 7:
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_cache_path,
                max_length=self.max_length,
                use_fast=True
            )
            self.vocab_size = cache_data.get('vocab_size', self.vocab_size)
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer cache: {str(e)}")
            return False

    def validate_tokenizer(self) -> bool:
        try:
            if not self.tokenizer:
                return False

            test_text = "Hello, world!"
            token_ids = self.encode(test_text)
            decoded_text = self.decode(token_ids)
            if not decoded_text:
                return False

            if len(self.tokenizer) > self.vocab_size:
                return False
            return True
        except Exception as e:
            logger.error(f"Tokenizer validation failed: {str(e)}")
            return False

    def get_vocab_size(self) -> int:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            return len(self.tokenizer)
        except Exception as e:
            logger.error(f"Failed to get vocabulary size: {str(e)}")
            return 0

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not texts:
                return {"input_ids": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}

            encoding = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }
        except Exception as e:
            logger.error(f"Failed to tokenize batch: {str(e)}")
            raise ValueError(f"Failed to tokenize batch: {str(e)}")

    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
                return []

            decoded_texts = self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return [text.strip() for text in decoded_texts]
        except Exception as e:
            logger.error(f"Failed to decode batch: {str(e)}")
            raise ValueError(f"Failed to decode batch: {str(e)}")

    def add_special_tokens(self, special_tokens: Dict[str, str]) -> None:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not isinstance(special_tokens, dict):
                raise ValueError("special_tokens must be a dictionary")

            self.tokenizer.add_special_tokens(special_tokens)
            self.vocab_size = len(self.tokenizer)
            self.save_cache()
        except Exception as e:
            logger.error(f"Failed to add special tokens: {str(e)}")
            raise ValueError(f"Failed to add special tokens: {str(e)}")

    def get_token_id(self, token: str) -> int:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")

            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                logger.warning(f"Token not found in vocabulary: {token}")
            return token_id
        except Exception as e:
            logger.error(f"Failed to get token ID: {str(e)}")
            return self.tokenizer.unk_token_id

    def get_token(self, token_id: int) -> str:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")

            return self.tokenizer.convert_ids_to_tokens(token_id)
        except Exception as e:
            logger.error(f"Failed to get token: {str(e)}")
            return ""

    def truncate_text(self, text: str) -> str:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not text:
                return ""

            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                text = self.tokenizer.convert_tokens_to_string(tokens)
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to truncate text: {str(e)}")
            return text

    def validate_text(self, text: str) -> bool:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            if not isinstance(text, str) or not text.strip():
                return False

            token_ids = self.encode(text)
            if token_ids.numel() == 0:
                return False

            decoded_text = self.decode(token_ids)
            if not decoded_text:
                return False
            return True
        except Exception as e:
            logger.error(f"Text validation failed: {str(e)}")
            return False

    def get_special_tokens(self) -> Dict[str, int]:
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")

            return {
                token: self.tokenizer.convert_tokens_to_ids(token)
                for token in self.tokenizer.all_special_tokens
            }
        except Exception as e:
            logger.error(f"Failed to get special tokens: {str(e)}")
            return {}

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        tokenizer = TranslationTokenizer(config_manager.config)
        if tokenizer.validate_tokenizer():
            test_text = "Hello, world!"
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            batch_tokens = tokenizer.tokenize_batch([test_text, "Good morning!"])
            decoded_batch = tokenizer.decode_batch(batch_tokens["input_ids"])
            special_tokens = tokenizer.get_special_tokens()
            tokenizer.save_cache()
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")