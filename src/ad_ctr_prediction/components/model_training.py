import os
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, precision_recall_curve, auc
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.utils.common import save_bin, save_json
from src.ad_ctr_prediction.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def load_split_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Split data file not found: {file_path}")

        logger.info(f"Loading split data from: {file_path}")
        return pd.read_csv(file_path)

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> XGBClassifier:
        target_col = self.config.target_column
        if target_col not in train_df.columns or target_col not in val_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train/validation data")

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        logger.info("Initializing XGBoost classifier")
        model = XGBClassifier(**self.config.model_params)

        logger.info("Starting XGBoost training with early stopping")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        logger.info("Model training completed")
        return model

    def evaluate_model(self, model: XGBClassifier, df: pd.DataFrame) -> dict:
        target_col = self.config.target_column
        X = df.drop(columns=[target_col])
        y = df[target_col]

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        
        precision, recall, _ = precision_recall_curve(y, y_proba) if y_proba is not None else (None, None, None)
        pr_auc = auc(recall, precision) if precision is not None and recall is not None else None

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc_score": roc_auc_score(y, y_proba) if y_proba is not None else None,
            "log_loss": log_loss(y, y_proba) if y_proba is not None else None,
            "pr_auc": pr_auc
        }

        return {k: float(v) for k, v in metrics.items()}

    def save_model(self, model: XGBClassifier) -> None:
        model_path = Path(self.config.model_file_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_bin(data=model, path=model_path)
        logger.info(f"Saved trained model at: {model_path}")

    def save_metrics(self, metrics: dict) -> None:
        metrics_path = Path(self.config.metrics_file_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path=metrics_path, data=metrics)
        logger.info(f"Saved training metrics at: {metrics_path}")

    def initiate_model_training(self) -> bool:
        try:
            train_df = self.load_split_data(self.config.train_file_path)
            val_df = self.load_split_data(self.config.validation_file_path)
            # test_df = self.load_split_data(self.config.test_file_path)

            model = self.train_model(train_df, val_df)
            self.save_model(model)

            logger.info("Evaluating model on validation data")
            validation_metrics = self.evaluate_model(model, val_df)
            # test_metrics = self.evaluate_model(model, test_df)

            metrics = {
                "validation": validation_metrics,
                # "test": test_metrics,
                "best_iteration": int(model.get_booster().best_iteration) if hasattr(model, "get_booster") and model.get_booster().best_iteration != -1 else None
            }

            self.save_metrics(metrics)
            logger.info("Model training stage completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise e