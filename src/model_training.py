import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from logger import get_logger

logger = get_logger(__name__)


def set_seeds(seed=42):
    """
    Set seeds for reproducibility across all libraries

    Args:
        seed (int): Seed value to be used across all random operations
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set environment variable for TensorFlow if it's used
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set scikit-learn's global random state if needed
    # Note: RandomForestRegressor already accepts random_state parameter

    logger.info(f"Random seeds set to {seed} for all libraries")


class ModelTraining:
    """
    Class responsible for training and evaluating regression models for ride demand prediction.

    Supports both RandomForest and XGBoost models.
    """

    def __init__(self, config):
        """
        Initialize the ModelTraining class with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing data paths and model parameters
        """
        artifact_dir = Path(config["data_ingestion"]["artifact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "validation.csv"

        self.model_training_config = config["model_training"]

    def load_data(self):
        """
        Load the training and validation datasets from CSV files.

        Returns:
            tuple: (train_data, val_data) as pandas DataFrames
        """
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path)
        return train_data, val_data

    def build_model(self, seed=None, model_type=None):
        """
        Build a regression model based on the specified type.

        Args:
            seed (int, optional): Random seed for model initialization. Defaults to None.
            model_type (str, optional): Type of model to build. Options: "random_forest" or "xgboost".
                                        If None, uses the value from config. Defaults to None.

        Returns:
            object: Initialized model instance
        """
        # Use provided seed if available, otherwise use from config
        if seed is None:
            seed = self.model_training_config["seed"]

        # Use provided model_type if available, otherwise use from config
        if model_type is None:
            model_type = self.model_training_config.get("model_type", "random_forest")

        if model_type == "random_forest":
            n_estimators = self.model_training_config["n_estimators"]
            max_samples = self.model_training_config["max_samples"]
            n_jobs = self.model_training_config["n_jobs"]

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_samples=max_samples,
                oob_score=root_mean_squared_error,
                n_jobs=n_jobs,
                random_state=seed,
            )

            logger.info("Building RandomForest model")

        elif model_type == "xgboost":
            # Get XGBoost parameters from config
            params = {
                "n_estimators": self.model_training_config.get("xgb_n_estimators", 100),
                "learning_rate": self.model_training_config.get("learning_rate", 0.1),
                "max_depth": self.model_training_config.get("max_depth", 6),
                "subsample": self.model_training_config.get("subsample", 0.8),
                "colsample_bytree": self.model_training_config.get(
                    "colsample_bytree", 0.8
                ),
                "reg_alpha": self.model_training_config.get("reg_alpha", 0),
                "reg_lambda": self.model_training_config.get("reg_lambda", 1),
                "random_state": seed,
                "n_jobs": self.model_training_config["n_jobs"],
            }

            model = xgb.XGBRegressor(**params)

            logger.info("Building XGBoost model")

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. Choose 'random_forest' or 'xgboost'"
            )

        return model

    def train_model(self, model, train_data):
        """
        Train the model on the provided training data.

        Args:
            model: Model instance to train
            train_data (DataFrame): Training data
        """
        X_train = train_data.drop(columns=["demand"])
        y_train = train_data["demand"]
        model.fit(X_train, y_train)

    def evaluate_model(self, model, val_data):
        """
        Evaluate the model on validation data.

        Args:
            model: Trained model instance
            val_data (DataFrame): Validation data

        Returns:
            float: Root Mean Squared Error (RMSE) of the model on validation data
        """
        X_val = val_data.drop(columns=["demand"])
        y_val = val_data["demand"]
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        # Only print oob_score for RandomForest which supports it
        if hasattr(model, "oob_score_"):
            logger.info(f"Out-Of-Bag Score: {model.oob_score_}")
            print(f"Out-Of-Bag Score: {model.oob_score_}")

        logger.info(f"RMSE: {rmse}")
        print(f"RMSE: {rmse}")
        return rmse

    def run(self, model_type=None):
        """
        Run the full model training and evaluation pipeline.

        Args:
            model_type (str, optional): Type of model to build and train.
                                       Options: "random_forest" or "xgboost".
                                       If None, uses the value from config.
                                       Defaults to None.
        """
        # Get model type from config if not provided
        if model_type is None:
            model_type = self.model_training_config.get("model_type", "random_forest")

        logger.info(f"Training {model_type} model")

        # Set seeds for reproducibility
        SEED = self.model_training_config["seed"]
        set_seeds(SEED)

        train_data, val_data = self.load_data()

        model = self.build_model(SEED, model_type=model_type)

        self.train_model(model, train_data)
        rmse = self.evaluate_model(model, val_data)

        logger.info(f"Model training completed. Final RMSE: {rmse}")
