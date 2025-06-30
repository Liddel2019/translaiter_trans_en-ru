import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from utils.config import ConfigManager
from collections import defaultdict
from datetime import datetime

# Configure logging for the dataset module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationDataset:
    """Manages loading, preprocessing, and batching of translation datasets for the translaiter_trans_en-ru project."""

    def __init__(self, config: Dict[str, Any], dataset_name: str):
        """
        Initialize the dataset with configuration and selected dataset name.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
            dataset_name (str): Name of the dataset to load (e.g., 'OPUS Tatoeba').
        """
        self.config = config
        self.config_manager = ConfigManager()
        self.dataset_name = dataset_name
        self.dataset_path = self.config_manager.get_absolute_path(
            self.config_manager.get_config_value("dataset.dataset_path", self.config, default="data/datasets")
        )
        self.cache_path = self.config_manager.get_absolute_path(
            self.config_manager.get_config_value("dataset.cache_path", self.config, default="data/cache")
        )
        self.max_length = self.config_manager.get_config_value("dataset.max_length", self.config, default=10)
        self.batch_size = self.config_manager.get_config_value("dataset.batch_size", self.config, default=32)
        self.clean_rules = self.config_manager.get_config_value("dataset.clean_rules", self.config, default=[])
        self.data = None
        self.processed_data = []
        self.batches = []
        self.is_cached = False

        # Validate dataset name
        available_datasets = self.config_manager.get_config_value("dataset.available_datasets", self.config, default=[])
        if not available_datasets:
            logger.error("No available datasets defined in configuration")
            raise ValueError("No available datasets defined in configuration")
        if dataset_name not in available_datasets:
            logger.error(f"Invalid dataset name: {dataset_name}. Available: {available_datasets}")
            raise ValueError(f"Dataset {dataset_name} not found in available datasets")

        # Initialize dataset
        self._initialize()
        logger.info(f"TranslationDataset initialized for {dataset_name}")

    def _initialize(self) -> None:
        """
        Helper method to initialize dataset by checking cache or loading data.
        """
        try:
            if self.load_cache():
                logger.info("Loaded dataset from cache")
                self.is_cached = True
            else:
                self.load_data()
                self.preprocess_data()
                self.save_cache()
                logger.info("Dataset loaded and cached")
        except Exception as e:
            logger.error(f"Dataset initialization failed: {str(e)}")
            raise

    def load_data(self) -> None:
        """
        Load raw data from the specified dataset path.
        """
        try:
            dataset_file = os.path.join(self.dataset_path, f"{self.dataset_name}.csv")
            if not os.path.exists(dataset_file):
                logger.error(f"Dataset file not found: {dataset_file}")
                raise FileNotFoundError(f"Dataset file {dataset_file} not found")

            # Load dataset using pandas (assuming CSV format with 'en' and 'ru' columns)
            self.data = pd.read_csv(dataset_file, encoding='utf-8')
            if 'en' not in self.data.columns or 'ru' not in self.data.columns:
                logger.error("Dataset missing required 'en' or 'ru' columns")
                raise ValueError("Invalid dataset format: missing 'en' or 'ru' columns")

            logger.info(f"Loaded {len(self.data)} samples from {dataset_file}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        """
        Preprocess data based on clean_rules and max_length.
        """
        try:
            if self.data is None:
                logger.error("No data to preprocess")
                raise ValueError("Data not loaded")

            self.processed_data = []
            for _, row in self.data.iterrows():
                en_text, ru_text = row['en'], row['ru']

                # Apply cleaning rules
                for rule in self.clean_rules:
                    en_text = re.sub(re.escape(rule), '', str(en_text))
                    ru_text = re.sub(re.escape(rule), '', str(ru_text))

                # Check length constraints
                en_words = en_text.split()
                ru_words = ru_text.split()
                if len(en_words) <= self.max_length and len(ru_words) <= self.max_length:
                    self.processed_data.append({
                        'en': en_text.strip(),
                        'ru': ru_text.strip()
                    })

            if not self.processed_data:
                logger.error("No valid data after preprocessing")
                raise ValueError("No valid data after preprocessing")

            logger.info(f"Preprocessed {len(self.processed_data)} samples")
            self.create_batches()
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def create_batches(self) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches of data for training.

        Returns:
            List[Dict[str, torch.Tensor]]: List of batched data.
        """
        try:
            if not self.processed_data:
                logger.error("No processed data to batch")
                raise ValueError("No processed data available")

            self.batches = []
            for i in range(0, len(self.processed_data), self.batch_size):
                batch_data = self.processed_data[i:i + self.batch_size]

                # Placeholder for tensor conversion (to be replaced with tokenizer integration)
                en_batch = [item['en'] for item in batch_data]
                ru_batch = [item['ru'] for item in batch_data]

                # Simulate tensor creation (actual tokenization will be in tokenizer.py)
                batch = {
                    'en': torch.tensor([len(text.split()) for text in en_batch], dtype=torch.long),
                    'ru': torch.tensor([len(text.split()) for text in ru_batch], dtype=torch.long),
                    'raw_en': en_batch,
                    'raw_ru': ru_batch
                }
                self.batches.append(batch)

            logger.info(f"Created {len(self.batches)} batches of size {self.batch_size}")
            return self.batches
        except Exception as e:
            logger.error(f"Batch creation failed: {str(e)}")
            raise

    def save_cache(self) -> None:
        """
        Save preprocessed data to the cache path.
        """
        try:
            os.makedirs(self.cache_path, exist_ok=True)
            cache_file = os.path.join(self.cache_path, f"{self.dataset_name}_cache.json")
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': self.dataset_name,
                'data': self.processed_data
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Preprocessed data cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
            raise

    def load_cache(self) -> bool:
        """
        Load cached data if available and valid.

        Returns:
            bool: True if cache loaded successfully, False otherwise.
        """
        try:
            cache_file = os.path.join(self.cache_path, f"{self.dataset_name}_cache.json")
            if not os.path.exists(cache_file):
                logger.info(f"No cache found at {cache_file}")
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if cache_data.get('dataset_name') != self.dataset_name:
                logger.warning("Cached dataset name mismatch")
                return False

            # Validate cache timestamp (e.g., not older than 7 days)
            cache_time = datetime.fromisoformat(cache_data.get('timestamp'))
            if (datetime.now() - cache_time).days > 7:
                logger.warning("Cache is outdated")
                return False

            self.processed_data = cache_data.get('data', [])
            if not self.processed_data:
                logger.warning("Empty cache data")
                return False

            self.create_batches()
            logger.info(f"Loaded {len(self.processed_data)} samples from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return False

    def validate_dataset(self) -> bool:
        """
        Validate the dataset integrity.

        Returns:
            bool: True if dataset is valid, False otherwise.
        """
        try:
            if self.data is None and not self.processed_data:
                logger.error("No data loaded for validation")
                return False

            # Check for non-empty data
            data_to_check = self.processed_data if self.processed_data else self.data.to_dict('records')
            if not data_to_check:
                logger.error("Dataset is empty")
                return False

            # Validate required fields and length constraints
            for item in data_to_check:
                en_text = item.get('en', '')
                ru_text = item.get('ru', '')
                if not en_text or not ru_text:
                    logger.error("Missing English or Russian text in dataset")
                    return False
                if len(en_text.split()) > self.max_length or len(ru_text.split()) > self.max_length:
                    logger.error(f"Text exceeds max_length ({self.max_length})")
                    return False

            logger.info("Dataset validation successful")
            return True
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False

    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: Sample containing English and Russian text.
        """
        try:
            if not self.processed_data:
                logger.error("No processed data available")
                raise ValueError("No processed data available")
            if index < 0 or index >= len(self.processed_data):
                logger.error(f"Invalid index: {index}")
                raise IndexError("Index out of range")

            sample = self.processed_data[index]
            logger.debug(f"Retrieved sample at index {index}: {sample}")
            return sample
        except Exception as e:
            logger.error(f"Failed to retrieve sample: {str(e)}")
            raise

    def get_batch(self, batch_index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a batch by index.

        Args:
            batch_index (int): Index of the batch to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Batch containing English and Russian data.
        """
        try:
            if not self.batches:
                logger.error("No batches available")
                raise ValueError("No batches created")
            if batch_index < 0 or batch_index >= len(self.batches):
                logger.error(f"Invalid batch index: {batch_index}")
                raise IndexError("Batch index out of range")

            batch = self.batches[batch_index]
            logger.debug(f"Retrieved batch {batch_index} with {len(batch['raw_en'])} samples")
            return batch
        except Exception as e:
            logger.error(f"Failed to retrieve batch: {str(e)}")
            raise

    def get_dataset_size(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        try:
            size = len(self.processed_data) if self.processed_data else len(self.data)
            logger.debug(f"Dataset size: {size}")
            return size
        except Exception as e:
            logger.error(f"Failed to get dataset size: {str(e)}")
            return 0

    def shuffle_data(self) -> None:
        """
        Shuffle the processed data to randomize sample order.
        """
        try:
            if not self.processed_data:
                logger.error("No processed data to shuffle")
                raise ValueError("No processed data available")

            np.random.shuffle(self.processed_data)
            self.create_batches()  # Recreate batches after shuffling
            logger.info("Dataset shuffled and batches recreated")
        except Exception as e:
            logger.error(f"Failed to shuffle data: {str(e)}")
            raise

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            train_ratio (float): Ratio for training set.
            val_ratio (float): Ratio for validation set.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing train, val, and test splits.
        """
        try:
            if not self.processed_data:
                logger.error("No processed data to split")
                raise ValueError("No processed data available")

            total_size = len(self.processed_data)
            train_ratio_config = self.config_manager.get_config_value("dataset.train_ratio", self.config, default=train_ratio)
            val_ratio_config = self.config_manager.get_config_value("dataset.val_ratio", self.config, default=val_ratio)
            train_size = int(total_size * train_ratio_config)
            val_size = int(total_size * val_ratio_config)
            test_size = total_size - train_size - val_size

            if train_size <= 0 or val_size <= 0 or test_size <= 0:
                logger.error("Invalid split ratios")
                raise ValueError("Invalid split ratios")

            # Shuffle before splitting
            np.random.shuffle(self.processed_data)

            splits = {
                'train': self.processed_data[:train_size],
                'val': self.processed_data[train_size:train_size + val_size],
                'test': self.processed_data[train_size + val_size:]
            }

            logger.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples")
            return splits
        except Exception as e:
            logger.error(f"Failed to split dataset: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Apply cleaning rules to a single text string.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        try:
            cleaned_text = str(text)
            for rule in self.clean_rules:
                cleaned_text = re.sub(re.escape(rule), '', cleaned_text)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Failed to clean text: {str(e)}")
            return text

    def validate_cache(self) -> bool:
        """
        Validate the integrity of the cached data.

        Returns:
            bool: True if cache is valid, False otherwise.
        """
        try:
            cache_file = os.path.join(self.cache_path, f"{self.dataset_name}_cache.json")
            if not os.path.exists(cache_file):
                logger.info("No cache file to validate")
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if not cache_data.get('data'):
                logger.warning("Cache data is empty")
                return False

            max_length = self.config_manager.get_config_value("dataset.max_length", self.config, default=10)
            for item in cache_data['data']:
                if 'en' not in item or 'ru' not in item:
                    logger.warning("Invalid cache entry: missing 'en' or 'ru'")
                    return False
                if len(item['en'].split()) > max_length or len(item['ru'].split()) > max_length:
                    logger.warning(f"Cache entry exceeds max_length ({max_length})")
                    return False

            logger.info("Cache validation successful")
            return True
        except Exception as e:
            logger.error(f"Cache validation failed: {str(e)}")
            return False

    def export_dataset(self, export_path: str) -> None:
        """
        Export the processed dataset to a specified path.

        Args:
            export_path (str): Path to export the dataset.
        """
        try:
            export_path = self.config_manager.get_absolute_path(export_path)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            export_data = pd.DataFrame(self.processed_data)
            export_data.to_csv(export_path, index=False, encoding='utf-8')
            logger.info(f"Dataset exported to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export dataset: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage for testing
    config_manager = ConfigManager()
    dataset = TranslationDataset(config_manager.config, "OPUS Tatoeba")
    if dataset.validate_dataset():
        dataset.shuffle_data()
        splits = dataset.split_dataset()
        sample = dataset.get_sample(0)
        logger.info(f"Sample: {sample}")
        dataset.export_dataset("data/exported_dataset.csv")