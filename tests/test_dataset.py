import unittest
import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import shutil
from data.dataset import TranslationDataset
from utils.config import ConfigManager

# Configure logging for the test module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tests/test_dataset.log")
    ]
)
logger = logging.getLogger(__name__)

class TestTranslationDataset(unittest.TestCase):
    """Unit tests for the TranslationDataset class in the translaiter_trans_en-ru project."""

    def setUp(self) -> None:
        """
        Set up the test environment by creating a temporary dataset file and initializing ConfigManager.

        Raises:
            RuntimeError: If setup fails due to file I/O or configuration errors.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.setUp()
        """
        try:
            self.config_manager = ConfigManager()
            self.test_dir = "tests/temp_datasets"
            self.test_cache_dir = "tests/temp_cache"
            self.test_dataset_name = "OPUS Tatoeba"
            self.test_dataset_path = os.path.join(self.test_dir, f"{self.test_dataset_name}.csv")
            self.test_cache_path = os.path.join(self.test_cache_dir, f"{self.test_dataset_name}_cache.json")

            # Create temporary directories
            os.makedirs(self.test_dir, exist_ok=True)
            os.makedirs(self.test_cache_dir, exist_ok=True)
            logger.info(f"Created temporary directories: {self.test_dir}, {self.test_cache_dir}")

            # Create a sample dataset
            self.create_test_dataset()
            logger.info("Test dataset file created successfully")

            # Prepare test configuration
            self.test_config = self.setup_test_config()
            logger.info("Test configuration initialized")

            # Verify configuration
            self.assertTrue(self.config_manager.validate_config(self.test_config), "Test configuration validation failed")
            logger.info("Test setup completed")
        except Exception as e:
            logger.error(f"Test setup failed: {str(e)}")
            raise RuntimeError(f"Test setup failed: {str(e)}")

    def tearDown(self) -> None:
        """
        Clean up the test environment by removing temporary files and directories.

        Raises:
            RuntimeError: If cleanup fails due to file I/O errors.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.tearDown()
        """
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                logger.info(f"Removed temporary dataset directory: {self.test_dir}")
            if os.path.exists(self.test_cache_dir):
                shutil.rmtree(self.test_cache_dir)
                logger.info(f"Removed temporary cache directory: {self.test_cache_dir}")
        except Exception as e:
            logger.error(f"Test cleanup failed: {str(e)}")
            raise RuntimeError(f"Test cleanup failed: {str(e)}")

    def setup_test_config(self) -> Dict[str, Any]:
        """
        Prepare a test configuration using ConfigManager.

        Returns:
            Dict[str, Any]: Test configuration dictionary.

        Raises:
            RuntimeError: If configuration setup fails.

        Example:
            >>> test = TestTranslationDataset()
            >>> config = test.setup_test_config()
        """
        try:
            config = self.config_manager.config.copy()
            config["dataset"]["dataset_path"] = self.test_dir
            config["dataset"]["cache_path"] = self.test_cache_dir
            config["dataset"]["available_datasets"] = [self.test_dataset_name]
            config["dataset"]["max_length"] = 5
            config["dataset"]["batch_size"] = 2
            config["dataset"]["clean_rules"] = ["<", ">"]
            logger.info("Test configuration prepared successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to setup test configuration: {str(e)}")
            raise RuntimeError(f"Test configuration setup failed: {str(e)}")

    def create_test_dataset(self) -> None:
        """
        Create a sample dataset CSV file for testing.

        Raises:
            RuntimeError: If dataset creation fails.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.create_test_dataset()
        """
        try:
            sample_data = {
                "en": [
                    "Hello world",
                    "This is a test",
                    "Good morning",
                    "Invalid <data>",
                    "Too long sentence here"
                ],
                "ru": [
                    "Привет, мир!",
                    "Это тест",
                    "Доброе утро",
                    "Недопустимые <данные>",
                    "Слишком длинное предложение здесь"
                ]
            }
            df = pd.DataFrame(sample_data)
            os.makedirs(self.test_dir, exist_ok=True)
            df.to_csv(self.test_dataset_path, index=False, encoding="utf-8")
            logger.info(f"Created test dataset file at {self.test_dataset_path} with {len(df)} samples")
        except Exception as e:
            logger.error(f"Failed to create test dataset: {str(e)}")
            raise RuntimeError(f"Failed to create test dataset: {str(e)}")

    def verify_dataset_integrity(self, dataset: TranslationDataset) -> None:
        """
        Verify the integrity of the dataset after initialization or processing.

        Args:
            dataset (TranslationDataset): The dataset instance to verify.

        Raises:
            AssertionError: If integrity checks fail.

        Example:
            >>> test = TestTranslationDataset()
            >>> dataset = TranslationDataset(test.test_config, "OPUS Tatoeba")
            >>> test.verify_dataset_integrity(dataset)
        """
        try:
            self.assertIsNotNone(dataset.data, "Dataset data is None")
            self.assertTrue(len(dataset.data) > 0, "Dataset is empty")
            self.assertTrue(dataset.validate_dataset(), "Dataset validation failed")
            self.assertTrue(len(dataset.processed_data) <= len(dataset.data), "Processed data size exceeds raw data")
            logger.info("Dataset integrity verification successful")
        except AssertionError as e:
            logger.error(f"Dataset integrity verification failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during integrity verification: {str(e)}")
            raise RuntimeError(f"Dataset integrity verification failed: {str(e)}")

    def test_init_valid_config(self) -> None:
        """
        Test dataset initialization with a valid configuration and dataset name.

        Raises:
            AssertionError: If initialization fails or dataset is not properly set up.
            RuntimeError: If unexpected errors occur during initialization.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_init_valid_config()
        """
        try:
            logger.info("Testing dataset initialization")
            dataset = TranslationDataset(self.config_manager.config, self.test_dataset_name)
            self.assertIsNotNone(dataset, "Dataset initialization returned None")
            self.assertEqual(dataset.dataset_name, self.test_dataset_name, "Dataset name mismatch")
            self.assertEqual(dataset.dataset_path, self.test_dir, "Dataset path mismatch")
            self.assertEqual(dataset.cache_path, self.test_cache_dir, "Cache path mismatch")
            self.assertEqual(dataset.max_length, 5, "Max length mismatch")
            self.assertEqual(dataset.batch_size, 2, "Batch size mismatch")
            self.verify_dataset_integrity(dataset)
            logger.info("Dataset initialized successfully with valid config")
        except AssertionError as e:
            logger.error(f"Initialization test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_init_valid_config: {str(e)}")
            raise RuntimeError(f"Initialization test failed: {str(e)}")

    def test_init_invalid_dataset(self) -> None:
        """
        Test dataset initialization with an invalid dataset name.

        Raises:
            AssertionError: If initialization does not raise ValueError for invalid dataset.
            RuntimeError: If unexpected errors occur during initialization.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_init_invalid_dataset()
        """
        try:
            logger.info("Testing initialization with invalid dataset name")
            with self.assertRaises(ValueError):
                dataset = TranslationDataset(self.test_config, "InvalidDataset")
                logger.error("Expected ValueError for invalid dataset name, but none raised")
            logger.info("Successfully caught ValueError for invalid dataset name")
        except AssertionError as e:
            logger.error(f"Invalid dataset test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_init_invalid_dataset: {str(e)}")
            raise RuntimeError(f"Invalid dataset test failed: {str(e)}")

    def test_load_data(self) -> None:
        """
        Test loading data from a sample CSV file using TranslationDataset.

        Raises:
            AssertionError: If data loading fails or data is invalid.
            RuntimeError: If unexpected errors occur during data loading.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_load_data()
        """
        try:
            logger.info("Testing data loading")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.load_data()
            self.assertIsNotNone(dataset.data, "Loaded data is None")
            self.assertEqual(len(dataset.data), 5, f"Expected 5 samples, got {len(dataset.data)}")
            self.assertTrue("en" in dataset.data.columns, "Missing 'en' column")
            self.assertTrue("ru" in dataset.data.columns, "Missing 'ru' column")
            logger.info("Data loaded successfully")
            self.verify_dataset_integrity(dataset)
        except AssertionError as e:
            logger.error(f"Data loading test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_load_data: {str(e)}")
            raise RuntimeError(f"Data loading test failed: {str(e)}")

    def test_load_data_missing_file(self) -> None:
        """
        Test loading data from a non-existent file.

        Raises:
            AssertionError: If loading does not raise FileNotFoundError.
            RuntimeError: If unexpected errors occur.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_load_data_missing_file()
        """
        try:
            logger.info("Testing data loading with missing file")
            os.rename(self.test_dataset_path, self.test_dataset_path + ".backup")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            with self.assertRaises(FileNotFoundError):
                dataset.load_data()
                logger.error("Expected FileNotFoundError, but none raised")
            logger.info("Successfully caught FileNotFoundError for missing dataset file")
            os.rename(self.test_dataset_path + ".backup", self.test_dataset_path)
        except AssertionError as e:
            logger.error(f"Missing file test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_load_data_missing_file: {str(e)}")
            raise RuntimeError(f"Missing file test failed: {str(e)}")

    def test_preprocess_data(self) -> None:
        """
        Test data preprocessing with clean_rules and max_length constraints.

        Raises:
            AssertionError: If preprocessing fails or output is invalid.
            RuntimeError: If unexpected errors occur during preprocessing.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_preprocess_data()
        """
        try:
            logger.info("Testing data preprocessing")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            self.assertTrue(len(dataset.processed_data) > 0, "No processed data after preprocessing")
            self.assertTrue(len(dataset.processed_data) <= len(dataset.data), "Processed data exceeds raw data")
            for item in dataset.processed_data:
                self.assertTrue(len(item["en"].split()) <= 5, "English text exceeds max_length")
                self.assertTrue(len(item["ru"].split()) <= 5, "Russian text exceeds max_length")
                self.assertFalse("<" in item["en"] and ">" in item["en"], "Cleaning rules not applied to English text")
                self.assertFalse("<" in item["ru"] and ">" in item["ru"], "Cleaning rules not applied to Russian text")
            logger.info(f"Preprocessing completed with {len(dataset.processed_data)} samples")
            self.verify_dataset_integrity(dataset)
        except AssertionError as e:
            logger.error(f"Preprocessing test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_preprocess_data: {str(e)}")
            raise RuntimeError(f"Preprocessing test failed: {str(e)}")

    def test_create_batches(self) -> None:
        """
        Test batch creation from processed data.

        Raises:
            AssertionError: If batch creation fails or batches are invalid.
            RuntimeError: If unexpected errors occur during batch creation.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_create_batches()
        """
        try:
            logger.info("Testing batch creation")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            batches = dataset.create_batches()
            self.assertTrue(len(batches) > 0, "No batches created")
            self.assertEqual(len(batches), (len(dataset.processed_data) + 1) // 2, "Incorrect number of batches")
            for batch in batches:
                self.assertTrue("en" in batch, "Missing 'en' key in batch")
                self.assertTrue("ru" in batch, "Missing 'ru' key in batch")
                self.assertTrue("raw_en" in batch, "Missing 'raw_en' key in batch")
                self.assertTrue("raw_ru" in batch, "Missing 'raw_ru' key in batch")
                self.assertTrue(len(batch["en"]) <= 2, "Batch size exceeds configured batch_size")
            logger.info(f"Created {len(batches)} batches successfully")
        except AssertionError as e:
            logger.error(f"Batch creation test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_create_batches: {str(e)}")
            raise RuntimeError(f"Batch creation test failed: {str(e)}")

    def test_validate_dataset(self) -> None:
        """
        Test dataset validation with valid data.

        Raises:
            AssertionError: If validation fails.
            RuntimeError: If unexpected errors occur during validation.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_validate_dataset()
        """
        try:
            logger.info("Testing dataset validation")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            self.assertTrue(dataset.validate_dataset(), "Dataset validation failed")
            logger.info("Validation test passed")
        except AssertionError as e:
            logger.error(f"Validation test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_validate_dataset: {str(e)}")
            raise RuntimeError(f"Validation test failed: {str(e)}")

    def test_validate_dataset_invalid(self) -> None:
        """
        Test dataset validation with invalid data (empty dataset).

        Raises:
            AssertionError: If validation does not fail for invalid data.
            RuntimeError: If unexpected errors occur.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_validate_dataset_invalid()
        """
        try:
            logger.info("Testing invalid dataset validation")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.data = None
            dataset.processed_data = []
            self.assertFalse(dataset.validate_dataset(), "Expected validation to fail for empty dataset")
            logger.info("Successfully validated failure for empty dataset")
        except AssertionError as e:
            logger.error(f"Invalid dataset validation test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_validate_dataset_invalid: {str(e)}")
            raise RuntimeError(f"Invalid dataset validation test failed: {str(e)}")

    def test_save_cache(self) -> None:
        """
        Test saving preprocessed data to cache.

        Raises:
            AssertionError: If cache saving fails or cache file is invalid.
            RuntimeError: If unexpected errors occur during cache saving.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_save_cache()
        """
        try:
            logger.info("Testing cache saving")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            dataset.save_cache()
            self.assertTrue(os.path.exists(self.test_cache_path), "Cache file not created")
            with open(self.test_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            self.assertEqual(cache_data["dataset_name"], self.test_dataset_name, "Cache dataset name mismatch")
            self.assertTrue(len(cache_data["data"]) > 0, "Cache data is empty")
            logger.info(f"Cache saved successfully at {self.test_cache_path}")
        except AssertionError as e:
            logger.error(f"Cache saving test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_save_cache: {str(e)}")
            raise RuntimeError(f"Cache saving test failed: {str(e)}")

    def test_load_cache(self) -> None:
        """
        Test loading data from cache.

        Raises:
            AssertionError: If cache loading fails or loaded data is invalid.
            RuntimeError: If unexpected errors occur during cache loading.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_load_cache()
        """
        try:
            logger.info("Testing cache loading")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            dataset.save_cache()
            dataset.processed_data = []  # Clear to simulate cache load
            self.assertTrue(dataset.load_cache(), "Cache loading failed")
            self.assertTrue(len(dataset.processed_data) > 0, "No data loaded from cache")
            self.verify_dataset_integrity(dataset)
            logger.info("Cache loaded successfully")
        except AssertionError as e:
            logger.error(f"Cache loading test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_load_cache: {str(e)}")
            raise RuntimeError(f"Cache loading test failed: {str(e)}")

    def test_load_cache_invalid(self) -> None:
        """
        Test loading an invalid cache file.

        Raises:
            AssertionError: If cache loading does not fail for invalid cache.
            RuntimeError: If unexpected errors occur.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_load_cache_invalid()
        """
        try:
            logger.info("Testing invalid cache loading")
            invalid_cache = {"dataset_name": "WrongDataset", "data": []}
            with open(self.test_cache_path, 'w', encoding='utf-8') as f:
                json.dump(invalid_cache, f)
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            self.assertFalse(dataset.load_cache(), "Expected cache loading to fail for invalid cache")
            logger.info("Successfully validated failure for invalid cache")
        except AssertionError as e:
            logger.error(f"Invalid cache test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_load_cache_invalid: {str(e)}")
            raise RuntimeError(f"Invalid cache test failed: {str(e)}")

    def test_shuffle_data(self) -> None:
        """
        Test shuffling of processed data.

        Raises:
            AssertionError: If data shuffling fails or order is unchanged.
            RuntimeError: If unexpected errors occur during shuffling.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_shuffle_data()
        """
        try:
            logger.info("Testing data shuffling")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            original_data = dataset.processed_data.copy()
            dataset.shuffle_data()
            self.assertTrue(len(dataset.processed_data) == len(original_data), "Data size changed after shuffling")
            self.assertFalse(dataset.processed_data == original_data, "Data order unchanged after shuffling")
            logger.info("Data shuffled successfully")
        except AssertionError as e:
            logger.error(f"Shuffle data test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_shuffle_data: {str(e)}")
            raise RuntimeError(f"Shuffle data test failed: {str(e)}")

    def test_split_dataset(self) -> None:
        """
        Test splitting dataset into train, val, and test sets.

        Raises:
            AssertionError: If dataset splitting fails or splits are invalid.
            RuntimeError: If unexpected errors occur during splitting.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_split_dataset()
        """
        try:
            logger.info("Testing dataset splitting")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            splits = dataset.split_dataset(train_ratio=0.6, val_ratio=0.2)
            self.assertTrue("train" in splits, "Missing train split")
            self.assertTrue("val" in splits, "Missing validation split")
            self.assertTrue("test" in splits, "Missing test split")
            total_samples = len(dataset.processed_data)
            expected_train = int(total_samples * 0.6)
            expected_val = int(total_samples * 0.2)
            expected_test = total_samples - expected_train - expected_val
            self.assertEqual(len(splits["train"]), expected_train, "Incorrect train split size")
            self.assertEqual(len(splits["val"]), expected_val, "Incorrect validation split size")
            self.assertEqual(len(splits["test"]), expected_test, "Incorrect test split size")
            logger.info(f"Dataset split successfully: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        except AssertionError as e:
            logger.error(f"Split dataset test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_split_dataset: {str(e)}")
            raise RuntimeError(f"Split dataset test failed: {str(e)}")

    def test_get_sample(self) -> None:
        """
        Test retrieving a sample by index.

        Raises:
            AssertionError: If sample retrieval fails or sample is invalid.
            RuntimeError: If unexpected errors occur during retrieval.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_get_sample()
        """
        try:
            logger.info("Testing sample retrieval")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            sample = dataset.get_sample(0)
            self.assertTrue("en" in sample, "Sample missing 'en' key")
            self.assertTrue("ru" in sample, "Sample missing 'ru' key")
            self.assertTrue(len(sample["en"].split()) <= 5, "Sample English text exceeds max_length")
            logger.info(f"Retrieved sample successfully: {sample}")
        except AssertionError as e:
            logger.error(f"Get sample test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_get_sample: {str(e)}")
            raise RuntimeError(f"Get sample test failed: {str(e)}")

    def test_get_batch(self) -> None:
        """
        Test retrieving a batch by index.

        Raises:
            AssertionError: If batch retrieval fails or batch is invalid.
            RuntimeError: If unexpected errors occur during retrieval.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_get_batch()
        """
        try:
            logger.info("Testing batch retrieval")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            batch = dataset.get_batch(0)
            self.assertTrue("en" in batch, "Batch missing 'en' key")
            self.assertTrue("ru" in batch, "Batch missing 'ru' key")
            self.assertTrue("raw_en" in batch, "Batch missing 'raw_en' key")
            self.assertTrue("raw_ru" in batch, "Batch missing 'raw_ru' key")
            self.assertTrue(len(batch["en"]) <= 2, "Batch size exceeds configured batch_size")
            logger.info(f"Retrieved batch successfully with {len(batch['raw_en'])} samples")
        except AssertionError as e:
            logger.error(f"Get batch test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_get_batch: {str(e)}")
            raise RuntimeError(f"Get batch test failed: {str(e)}")

    def test_export_dataset(self) -> None:
        """
        Test exporting the processed dataset to a specified path.

        Raises:
            AssertionError: If dataset export fails or exported file is invalid.
            RuntimeError: If unexpected errors occur during export.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_export_dataset()
        """
        try:
            logger.info("Testing dataset export")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            dataset.preprocess_data()
            export_path = "tests/exported_dataset.csv"
            dataset.export_dataset(export_path)
            self.assertTrue(os.path.exists(export_path), "Exported dataset file not created")
            exported_data = pd.read_csv(export_path, encoding='utf-8')
            self.assertEqual(len(exported_data), len(dataset.processed_data), "Exported data size mismatch")
            self.assertTrue("en" in exported_data.columns, "Missing 'en' column in exported data")
            self.assertTrue("ru" in exported_data.columns, "Missing 'ru' column in exported data")
            logger.info(f"Dataset exported successfully to {export_path}")
        except AssertionError as e:
            logger.error(f"Export dataset test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_export_dataset: {str(e)}")
            raise RuntimeError(f"Export dataset test failed: {str(e)}")

    def test_clean_text(self) -> None:
        """
        Test cleaning text with clean_rules.

        Raises:
            AssertionError: If text cleaning fails or output is invalid.
            RuntimeError: If unexpected errors occur during cleaning.

        Example:
            >>> test = TestTranslationDataset()
            >>> test.test_clean_text()
        """
        try:
            logger.info("Testing text cleaning")
            dataset = TranslationDataset(self.test_config, self.test_dataset_name)
            test_text = "Hello <world> test"
            cleaned_text = dataset.clean_text(test_text)
            self.assertFalse("<" in cleaned_text, "Cleaning rule '<' not applied")
            self.assertFalse(">" in cleaned_text, "Cleaning rule '>' not applied")
            self.assertEqual(cleaned_text, "Hello world test", "Unexpected cleaned text output")
            logger.info("Text cleaned successfully")
        except AssertionError as e:
            logger.error(f"Clean text test failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in test_clean_text: {str(e)}")
            raise RuntimeError(f"Clean text test failed: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("Starting TranslationDataset unit tests")
        unittest.main(verbosity=2)
        logger.info("TranslationDataset unit tests completed successfully")
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise