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
from data.tokenizer import TranslationTokenizer  # Import the tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationDataset:
    def __init__(self, config: Dict[str, Any], dataset_name: str):
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
        self.tokenizer = TranslationTokenizer(config)  # Initialize tokenizer

        available_datasets = self.config_manager.get_config_value("dataset.available_datasets", self.config, default=[])
        if not available_datasets:
            logger.error("No available datasets defined in configuration")
            raise ValueError("No available datasets defined in configuration")
        if dataset_name not in available_datasets:
            logger.error(f"Invalid dataset name: {dataset_name}. Available: {available_datasets}")
            raise ValueError(f"Dataset {dataset_name} not found in available datasets")

        self._initialize()

    def _initialize(self) -> None:
        try:
            if self.load_cache():
                self.is_cached = True
            else:
                self.load_data()
                self.preprocess_data()
                self.save_cache()
        except Exception as e:
            logger.error(f"Dataset initialization failed: {str(e)}")
            raise

    def load_data(self) -> None:
        try:
            dataset_file = os.path.join(self.dataset_path, f"{self.dataset_name}.csv")
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file {dataset_file} not found")

            self.data = pd.read_csv(dataset_file, encoding='utf-8')
            if 'en' not in self.data.columns or 'ru' not in self.data.columns:
                raise ValueError("Invalid dataset format: missing 'en' or 'ru' columns")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        try:
            if self.data is None:
                raise ValueError("Data not loaded")

            self.processed_data = []
            for _, row in self.data.iterrows():
                en_text, ru_text = row['en'], row['ru']
                for rule in self.clean_rules:
                    en_text = re.sub(re.escape(rule), '', str(en_text))
                    ru_text = re.sub(re.escape(rule), '', str(ru_text))
                en_words = en_text.split()
                ru_words = ru_text.split()
                if len(en_words) <= self.max_length and len(ru_words) <= self.max_length:
                    self.processed_data.append({
                        'en': en_text.strip(),
                        'ru': ru_text.strip()
                    })

            if not self.processed_data:
                raise ValueError("No valid data after preprocessing")
            self.create_batches()
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def create_batches(self) -> List[Dict[str, torch.Tensor]]:
        try:
            if not self.processed_data:
                raise ValueError("No processed data available")

            self.batches = []
            for i in range(0, len(self.processed_data), self.batch_size):
                batch_data = self.processed_data[i:i + self.batch_size]
                en_batch = [item['en'] for item in batch_data]
                ru_batch = [item['ru'] for item in batch_data]

                # Tokenize batches using tokenize_batch
                en_encoding = self.tokenizer.tokenize_batch(en_batch)
                ru_encoding = self.tokenizer.tokenize_batch(ru_batch)

                batch = {
                    'en': en_encoding["input_ids"],  # Shape: [batch_size, max_length]
                    'ru': ru_encoding["input_ids"],  # Shape: [batch_size, max_length]
                    'raw_en': en_batch,
                    'raw_ru': ru_batch
                }
                self.batches.append(batch)
            return self.batches
        except Exception as e:
            logger.error(f"Batch creation failed: {str(e)}")
            raise

    def save_cache(self) -> None:
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
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
            raise

    def load_cache(self) -> bool:
        try:
            cache_file = os.path.join(self.cache_path, f"{self.dataset_name}_cache.json")
            if not os.path.exists(cache_file):
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if cache_data.get('dataset_name') != self.dataset_name:
                return False

            cache_time = datetime.fromisoformat(cache_data.get('timestamp'))
            if (datetime.now() - cache_time).days > 7:
                return False

            self.processed_data = cache_data.get('data', [])
            if not self.processed_data:
                return False

            self.create_batches()
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return False

    def validate_dataset(self) -> bool:
        try:
            if self.data is None and not self.processed_data:
                return False

            data_to_check = self.processed_data if self.processed_data else self.data.to_dict('records')
            if not data_to_check:
                return False

            for item in data_to_check:
                en_text = item.get('en', '')
                ru_text = item.get('ru', '')
                if not en_text or not ru_text:
                    return False
                if len(en_text.split()) > self.max_length or len(ru_text.split()) > self.max_length:
                    return False
            return True
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False

    def get_sample(self, index: int) -> Dict[str, Any]:
        try:
            if not self.processed_data:
                raise ValueError("No processed data available")
            if index < 0 or index >= len(self.processed_data):
                raise IndexError("Index out of range")
            return self.processed_data[index]
        except Exception as e:
            logger.error(f"Failed to retrieve sample: {str(e)}")
            raise

    def get_batch(self, batch_index: int) -> Dict[str, torch.Tensor]:
        try:
            if not self.batches:
                raise ValueError("No batches created")
            if batch_index < 0 or batch_index >= len(self.batches):
                raise IndexError("Batch index out of range")
            return self.batches[batch_index]
        except Exception as e:
            logger.error(f"Failed to retrieve batch: {str(e)}")
            raise

    def get_dataset_size(self) -> int:
        try:
            return len(self.processed_data) if self.processed_data else len(self.data)
        except Exception as e:
            logger.error(f"Failed to get dataset size: {str(e)}")
            return 0

    def shuffle_data(self) -> None:
        try:
            if not self.processed_data:
                raise ValueError("No processed data available")
            np.random.shuffle(self.processed_data)
            self.create_batches()
        except Exception as e:
            logger.error(f"Failed to shuffle data: {str(e)}")
            raise

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        try:
            if not self.processed_data:
                raise ValueError("No processed data available")

            total_size = len(self.processed_data)
            train_ratio_config = self.config_manager.get_config_value("dataset.train_ratio", self.config, default=train_ratio)
            val_ratio_config = self.config_manager.get_config_value("dataset.val_ratio", self.config, default=val_ratio)
            train_size = int(total_size * train_ratio_config)
            val_size = int(total_size * val_ratio_config)
            test_size = total_size - train_size - val_size

            if train_size <= 0 or val_size <= 0 or test_size <= 0:
                raise ValueError("Invalid split ratios")

            np.random.shuffle(self.processed_data)
            splits = {
                'train': self.processed_data[:train_size],
                'val': self.processed_data[train_size:train_size + val_size],
                'test': self.processed_data[train_size + val_size:]
            }
            return splits
        except Exception as e:
            logger.error(f"Failed to split dataset: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        try:
            cleaned_text = str(text)
            for rule in self.clean_rules:
                cleaned_text = re.sub(re.escape(rule), '', cleaned_text)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Failed to clean text: {str(e)}")
            return text

    def validate_cache(self) -> bool:
        try:
            cache_file = os.path.join(self.cache_path, f"{self.dataset_name}_cache.json")
            if not os.path.exists(cache_file):
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if not cache_data.get('data'):
                return False

            max_length = self.config_manager.get_config_value("dataset.max_length", self.config, default=10)
            for item in cache_data['data']:
                if 'en' not in item or 'ru' not in item:
                    return False
                if len(item['en'].split()) > max_length or len(item['ru'].split()) > max_length:
                    return False
            return True
        except Exception as e:
            logger.error(f"Cache validation failed: {str(e)}")
            return False

    def export_dataset(self, export_path: str) -> None:
        try:
            export_path = self.config_manager.get_absolute_path(export_path)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            export_data = pd.DataFrame(self.processed_data)
            export_data.to_csv(export_path, index=False, encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to export dataset: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        dataset = TranslationDataset(config_manager.config, "OPUS Tatoeba")
        if dataset.validate_dataset():
            dataset.shuffle_data()
            splits = dataset.split_dataset()
            sample = dataset.get_sample(0)
            dataset.export_dataset("data/exported_dataset.csv")
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")