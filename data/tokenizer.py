import os
import logging
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
from transformers import AutoTokenizer
from utils.config import ConfigManager

# Configure logging for the tokenizer module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationTokenizer:
    """Handles tokenization for the translaiter_trans_en-ru project using a pre-trained tokenizer."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tokenizer with configuration settings using ConfigManager.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.

        Raises:
            ValueError: If any configuration value is invalid (e.g., empty tokenizer_name, non-positive max_length).

        Example:
            >>> config_manager = ConfigManager()
            >>> tokenizer = TranslationTokenizer(config_manager.config)
            >>> tokenizer.validate_tokenizer()
            True
        """
        self.config = config
        self.config_manager = ConfigManager()
        self.tokenizer = None
        self.is_cached = False

        # Fetch configuration values using ConfigManager
        try:
            self.tokenizer_name = self.config_manager.get_config_value(
                "tokenizer.tokenizer_name", self.config, default="Helsinki-NLP/opus-mt-en-ru"
            )
            logger.info(f"Tokenizer name loaded: {self.tokenizer_name}")
            self.validate_config_value("tokenizer_name", self.tokenizer_name, str, non_empty=True)

            self.tokenizer_cache_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "tokenizer.tokenizer_cache_path", self.config, default="data/tokenizer_cache"
                )
            )
            logger.info(f"Tokenizer cache path loaded: {self.tokenizer_cache_path}")
            self.validate_config_value("tokenizer_cache_path", self.tokenizer_cache_path, str, non_empty=True)

            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", self.config, default=10
            )
            logger.info(f"Max length loaded: {self.max_length}")
            self.validate_config_value("max_length", self.max_length, int, positive=True)

            self.vocab_size = self.config_manager.get_config_value(
                "tokenizer.vocab_size", self.config, default=32000
            )
            logger.info(f"Vocabulary size loaded: {self.vocab_size}")
            self.validate_config_value("vocab_size", self.vocab_size, int, positive=True)

            # Initialize tokenizer
            self._initialize()
            logger.info(f"TranslationTokenizer initialized with tokenizer: {self.tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer configuration: {str(e)}")
            raise ValueError(f"Tokenizer initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False, positive: bool = False) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key for logging purposes.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string values are non-empty.
            positive (bool): If True, ensures numeric values are positive.

        Raises:
            ValueError: If the value does not meet validation criteria.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> tokenizer.validate_config_value("max_length", 10, int, positive=True)
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
            logger.debug(f"Validated {key}: {value}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def _initialize(self) -> None:
        """
        Helper method to initialize the tokenizer by checking cache or loading the pre-trained model.

        Raises:
            ValueError: If tokenizer initialization or validation fails.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> tokenizer._initialize()
        """
        try:
            # Re-fetch config values for consistency
            self.tokenizer_name = self.config_manager.get_config_value(
                "tokenizer.tokenizer_name", self.config, default="Helsinki-NLP/opus-mt-en-ru"
            )
            logger.info(f"Re-fetched tokenizer name: {self.tokenizer_name}")

            self.tokenizer_cache_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "tokenizer.tokenizer_cache_path", self.config, default="data/tokenizer_cache"
                )
            )
            logger.info(f"Re-fetched tokenizer cache path: {self.tokenizer_cache_path}")

            self.max_length = self.config_manager.get_config_value(
                "tokenizer.max_length", self.config, default=10
            )
            logger.info(f"Re-fetched max length: {self.max_length}")

            self.vocab_size = self.config_manager.get_config_value(
                "tokenizer.vocab_size", self.config, default=32000
            )
            logger.info(f"Re-fetched vocabulary size: {self.vocab_size}")

            if self.load_cache():
                logger.info("Tokenizer loaded from cache")
                self.is_cached = True
            else:
                self.load_tokenizer()
                self.save_cache()
                logger.info("Tokenizer loaded and cached")
            if not self.validate_tokenizer():
                logger.error("Tokenizer validation failed")
                raise ValueError("Invalid tokenizer configuration")
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

    def load_tokenizer(self) -> None:
        """
        Load the pre-trained tokenizer from the specified name.

        Raises:
            ValueError: If tokenizer loading fails.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> tokenizer.load_tokenizer()
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=self.tokenizer_cache_path,
                max_length=self.max_length,
                use_fast=True
            )
            actual_vocab_size = len(self.tokenizer)
            if actual_vocab_size != self.vocab_size:
                logger.warning(
                    f"Tokenizer vocabulary size ({actual_vocab_size}) differs from configured size ({self.vocab_size})"
                )
                self.vocab_size = actual_vocab_size
            logger.info(f"Loaded tokenizer: {self.tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_name}: {str(e)}")
            raise ValueError(f"Failed to load tokenizer: {str(e)}")

    def encode(self, text: str) -> torch.Tensor:
        """
        Convert text to token IDs.

        Args:
            text (str): Input text to encode.

        Returns:
            torch.Tensor: Tensor of token IDs.

        Raises:
            ValueError: If tokenizer is not initialized or text is invalid.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> token_ids = tokenizer.encode("Hello, world!")
            >>> print(token_ids)
            tensor([2, 123, 456, ..., 0])
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not isinstance(text, str) or not text.strip():
                logger.warning("Invalid or empty input text for encoding")
                return torch.tensor([], dtype=torch.long)

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            token_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
            logger.debug(f"Encoded text: {text[:50]}... to {len(token_ids)} tokens")
            return token_ids
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise ValueError(f"Failed to encode text: {str(e)}")

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs.

        Returns:
            str: Decoded text.

        Raises:
            ValueError: If tokenizer is not initialized or token_ids is invalid.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> text = tokenizer.decode(torch.tensor([2, 123, 456, ..., 0]))
            >>> print(text)
            'Hello, world!'
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
                logger.warning("Invalid or empty token IDs for decoding")
                return ""

            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()

            decoded_text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            logger.debug(f"Decoded {len(token_ids)} tokens to text: {decoded_text[:50]}...")
            return decoded_text.strip()
        except Exception as e:
            logger.error(f"Failed to decode token IDs: {str(e)}")
            raise ValueError(f"Failed to decode token IDs: {str(e)}")

    def save_cache(self) -> None:
        """
        Save the tokenizer state to the cache path.

        Raises:
            ValueError: If saving the cache fails.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> tokenizer.save_cache()
        """
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
            logger.info(f"Tokenizer state cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer cache: {str(e)}")
            raise ValueError(f"Failed to save tokenizer cache: {str(e)}")

    def load_cache(self) -> bool:
        """
        Load cached tokenizer state if available and valid.

        Returns:
            bool: True if cache loaded successfully, False otherwise.

        Raises:
            ValueError: If loading the cache fails critically.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> loaded = tokenizer.load_cache()
            >>> print(loaded)
            True
        """
        try:
            cache_file = os.path.join(self.tokenizer_cache_path, f"{self.tokenizer_name.replace('/', '_')}_cache.json")
            if not os.path.exists(cache_file):
                logger.info(f"No tokenizer cache found at {cache_file}")
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if cache_data.get('tokenizer_name') != self.tokenizer_name:
                logger.warning("Cached tokenizer name mismatch")
                return False

            cache_time = datetime.fromisoformat(cache_data.get('timestamp'))
            if (datetime.now() - cache_time).days > 7:
                logger.warning("Tokenizer cache is outdated")
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_cache_path,
                max_length=self.max_length,
                use_fast=True
            )
            self.vocab_size = cache_data.get('vocab_size', self.vocab_size)
            logger.info(f"Loaded tokenizer from cache: {self.tokenizer_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer cache: {str(e)}")
            return False

    def validate_tokenizer(self) -> bool:
        """
        Validate the tokenizer integrity.

        Returns:
            bool: True if tokenizer is valid, False otherwise.

        Raises:
            ValueError: If critical validation checks fail.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> valid = tokenizer.validate_tokenizer()
            >>> print(valid)
            True
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                return False

            test_text = "Hello, world!"
            token_ids = self.encode(test_text)
            decoded_text = self.decode(token_ids)
            if not decoded_text:
                logger.error("Tokenizer failed to decode sample text")
                return False

            if len(self.tokenizer) > self.vocab_size:
                logger.warning(f"Tokenizer vocabulary size ({len(self.tokenizer)}) exceeds configured size ({self.vocab_size})")
                return False

            logger.info("Tokenizer validation successful")
            return True
        except Exception as e:
            logger.error(f"Tokenizer validation failed: {str(e)}")
            return False

    def get_vocab_size(self) -> int:
        """
        Retrieve the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> vocab_size = tokenizer.get_vocab_size()
            >>> print(vocab_size)
            32000
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")
            vocab_size = len(self.tokenizer)
            logger.debug(f"Vocabulary size: {vocab_size}")
            return vocab_size
        except Exception as e:
            logger.error(f"Failed to get vocabulary size: {str(e)}")
            return 0

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.

        Args:
            texts (List[str]): List of texts to tokenize.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing token IDs and attention masks.

        Raises:
            ValueError: If tokenizer is not initialized or texts are invalid.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> batch = tokenizer.tokenize_batch(["Hello!", "Good morning!"])
            >>> print(batch["input_ids"].shape)
            torch.Size([2, 10])
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not texts:
                logger.warning("Empty text list for batch tokenization")
                return {"input_ids": torch.tensor([], dtype=torch.long), "attention_mask": torch.tensor([], dtype=torch.long)}

            encoding = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            logger.debug(f"Tokenized batch of {len(texts)} texts")
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }
        except Exception as e:
            logger.error(f"Failed to tokenize batch: {str(e)}")
            raise ValueError(f"Failed to tokenize batch: {str(e)}")

    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode a batch of token IDs to texts.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs (batch_size, seq_length).

        Returns:
            List[str]: List of decoded texts.

        Raises:
            ValueError: If tokenizer is not initialized or token_ids is invalid.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> texts = tokenizer.decode_batch(torch.tensor([[2, 123, 456], [2, 789, 101]]))
            >>> print(texts)
            ['Hello', 'Good']
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
                logger.warning("Invalid or empty token IDs for batch decoding")
                return []

            decoded_texts = self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            logger.debug(f"Decoded batch of {len(decoded_texts)} texts")
            return [text.strip() for text in decoded_texts]
        except Exception as e:
            logger.error(f"Failed to decode batch: {str(e)}")
            raise ValueError(f"Failed to decode batch: {str(e)}")

    def add_special_tokens(self, special_tokens: Dict[str, str]) -> None:
        """
        Add special tokens to the tokenizer.

        Args:
            special_tokens (Dict[str, str]): Dictionary of special token names and their values.

        Raises:
            ValueError: If tokenizer is not initialized or special_tokens is invalid.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> tokenizer.add_special_tokens({'<CUSTOM>': '<CUSTOM>'})
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not isinstance(special_tokens, dict):
                logger.error("special_tokens must be a dictionary")
                raise ValueError("special_tokens must be a dictionary")

            self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added special tokens: {list(special_tokens.keys())}")
            self.vocab_size = len(self.tokenizer)
            self.save_cache()
        except Exception as e:
            logger.error(f"Failed to add special tokens: {str(e)}")
            raise ValueError(f"Failed to add special tokens: {str(e)}")

    def get_token_id(self, token: str) -> int:
        """
        Get the token ID for a given token.

        Args:
            token (str): Token to look up.

        Returns:
            int: Token ID.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> token_id = tokenizer.get_token_id("hello")
            >>> print(token_id)
            123
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                logger.warning(f"Token not found in vocabulary: {token}")
            logger.debug(f"Token {token} mapped to ID {token_id}")
            return token_id
        except Exception as e:
            logger.error(f"Failed to get token ID: {str(e)}")
            return self.tokenizer.unk_token_id

    def get_token(self, token_id: int) -> str:
        """
        Get the token string for a given token ID.

        Args:
            token_id (int): Token ID to look up.

        Returns:
            str: Token string.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> token = tokenizer.get_token(123)
            >>> print(token)
            'hello'
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            token = self.tokenizer.convert_ids_to_tokens(token_id)
            logger.debug(f"Token ID {token_id} mapped to token {token}")
            return token
        except Exception as e:
            logger.error(f"Failed to get token: {str(e)}")
            return ""

    def truncate_text(self, text: str) -> str:
        """
        Truncate text to the configured max_length.

        Args:
            text (str): Input text to truncate.

        Returns:
            str: Truncated text.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> truncated = tokenizer.truncate_text("This is a very long sentence")
            >>> print(truncated)
            'This is a very'
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not text:
                logger.warning("Empty text for truncation")
                return ""

            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                text = self.tokenizer.convert_tokens_to_string(tokens)
                logger.debug(f"Truncated text to {self.max_length} tokens")
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to truncate text: {str(e)}")
            return text

    def validate_text(self, text: str) -> bool:
        """
        Validate if the text can be tokenized correctly.

        Args:
            text (str): Input text to validate.

        Returns:
            bool: True if text is valid, False otherwise.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> valid = tokenizer.validate_text("Hello, world!")
            >>> print(valid)
            True
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            if not isinstance(text, str) or not text.strip():
                logger.warning("Invalid or empty text for validation")
                return False

            token_ids = self.encode(text)
            if token_ids.numel() == 0:
                logger.warning("Text produced empty tokenization")
                return False

            decoded_text = self.decode(token_ids)
            if not decoded_text:
                logger.warning("Text failed to decode")
                return False

            logger.debug("Text validation successful")
            return True
        except Exception as e:
            logger.error(f"Text validation failed: {str(e)}")
            return False

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Retrieve all special tokens and their IDs.

        Returns:
            Dict[str, int]: Dictionary mapping special tokens to their IDs.

        Raises:
            ValueError: If tokenizer is not initialized.

        Example:
            >>> tokenizer = TranslationTokenizer(config)
            >>> special_tokens = tokenizer.get_special_tokens()
            >>> print(special_tokens)
            {'<pad>': 0, '</s>': 2, ...}
        """
        try:
            if not self.tokenizer:
                logger.error("Tokenizer not initialized")
                raise ValueError("Tokenizer not initialized")

            special_tokens = {
                token: self.tokenizer.convert_tokens_to_ids(token)
                for token in self.tokenizer.all_special_tokens
            }
            logger.debug(f"Retrieved special tokens: {list(special_tokens.keys())}")
            return special_tokens
        except Exception as e:
            logger.error(f"Failed to get special tokens: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage for testing
    try:
        config_manager = ConfigManager()
        tokenizer = TranslationTokenizer(config_manager.config)
        logger.info("Configuration values:")
        logger.info(f"Tokenizer name: {tokenizer.tokenizer_name}")
        logger.info(f"Tokenizer cache path: {tokenizer.tokenizer_cache_path}")
        logger.info(f"Max length: {tokenizer.max_length}")
        logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
        if tokenizer.validate_tokenizer():
            test_text = "Hello, world!"
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            logger.info(f"Encoded and decoded: {test_text} -> {decoded_text}")
            batch_tokens = tokenizer.tokenize_batch([test_text, "Good morning!"])
            decoded_batch = tokenizer.decode_batch(batch_tokens["input_ids"])
            logger.info(f"Batch decoded: {decoded_batch}")
            special_tokens = tokenizer.get_special_tokens()
            logger.info(f"Special tokens: {special_tokens}")
            tokenizer.save_cache()
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")