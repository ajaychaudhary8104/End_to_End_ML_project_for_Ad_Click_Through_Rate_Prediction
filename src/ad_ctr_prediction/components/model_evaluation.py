import os
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report, roc_auc_score, log_loss
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.ad_ctr_prediction.utils.common import save_json
from pathlib import Path
from src.ad_ctr_prediction import logger
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from src.ad_ctr_prediction.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Object with numpy types converted to Python types
        """
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load test data from CSV file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from: {data_path}")
        return pd.read_csv(data_path)

    def load_model(self, model_path: str):
        """Load the trained model from pickle file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def eval_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: Dictionary containing all metrics
        """
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        pr_precision, re_recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(re_recall, pr_precision)
        
        # For binary or multi-class with proper handling
        try:
            roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
        except:
            roc_auc = 0.0
            logger.warning("Could not calculate ROC-AUC score")
        
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "log_loss": float(logloss),
            "pr_auc": float(pr_auc)
        }
        
        return metrics

    def create_confusion_matrix_artifact(self, y_true, y_pred):
        """Create confusion matrix and return as dict."""
        cm = confusion_matrix(y_true, y_pred)
        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "labels": [int(label) for label in set(y_true)]  # Convert numpy int64 to Python int
        }
        return cm_dict

    def log_into_mlflow(self):
        """
        Log model evaluation metrics and model to MLflow with versioning.
        """
        try:
            # Load test data
            test_data = self.load_data(str(self.config.test_data_path))
            logger.info(f"Test data shape: {test_data.shape}")
            
            # Load trained model
            model = self.load_model(str(self.config.model_path))
            logger.info("Model loaded successfully")
            
            # Prepare features and target
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]
            
            logger.info(f"Features shape: {test_x.shape}")
            logger.info(f"Target shape: {test_y.shape}")
            
            # Set MLflow tracking URI for remote logging
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            logger.info(f"MLflow tracking URI: {self.config.mlflow_uri}")
            logger.info(f"Tracking URL type: {tracking_url_type_store}")
            
            # Start MLflow run
            with mlflow.start_run():
                # Make predictions
                predicted_qualities = model.predict(test_x)
                logger.info(f"Predictions made. Unique predictions: {set(predicted_qualities)}")
                y_prob = model.predict_proba(test_x)[:, 1] if hasattr(model, "predict_proba") else None
                logger.info(f"Predicted probabilities shape: {y_prob.shape if y_prob is not None else 'N/A'}")

                # Calculate metrics
                metrics = self.eval_metrics(test_y.values.flatten(), predicted_qualities)
                logger.info(f"Metrics calculated: {metrics}")
                
                # Create confusion matrix
                cm_artifact = self.create_confusion_matrix_artifact(
                    test_y.values.flatten(), 
                    predicted_qualities
                )
                logger.info(f"Confusion matrix: {cm_artifact}")
                
                # Create classification report
                class_report = classification_report(
                    test_y.values.flatten(), 
                    predicted_qualities, 
                    output_dict=True,
                    zero_division=0
                )
                logger.info("Classification report generated")
                
                # Convert numpy types to Python types for JSON serialization
                class_report = self._convert_numpy_types(class_report)
                cm_artifact = self._convert_numpy_types(cm_artifact)
                
                # Combine all metrics for saving
                all_metrics = {
                    **metrics,
                    "confusion_matrix": cm_artifact["confusion_matrix"],
                    "classification_report": class_report
                }
                
                # Save metrics locally
                logger.info(f"Saving metrics to: {self.config.metric_file_name}")
                save_json(path=Path(self.config.metric_file_name), data=all_metrics)
                
                # Log parameters to MLflow
                logger.info("Logging parameters to MLflow")
                mlflow.log_params(self.config.all_params)
                
                # Log metrics to MLflow
                logger.info("Logging metrics to MLflow")
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                    logger.info(f"Logged {metric_name}: {metric_value}")
                
                # Log confusion matrix
                mlflow.log_dict(cm_artifact, "confusion_matrix.json")
                logger.info("Logged confusion matrix artifact")
                
                # Log classification report
                mlflow.log_dict(class_report, "classification_report.json")
                logger.info("Logged classification report artifact")
                
                # Register model with versioning
                if tracking_url_type_store != "file":
                    logger.info("Registering model to MLflow Model Registry")
                    try:
                        mlflow.sklearn.log_model(
                            model, 
                            "model", 
                            registered_model_name="XGBClassifier"
                        )
                        logger.info("Model registered successfully with versioning")
                    except Exception as e:
                        logger.warning(f"Could not register model to registry: {str(e)}")
                        logger.info("Logging model locally instead")
                        mlflow.sklearn.log_model(model, "model")
                else:
                    logger.info("Using local file store, logging model locally")
                    mlflow.sklearn.log_model(model, "model")
                
                logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise e

    def save_evaluation_results(self):
        """
        Load test data, evaluate model, and save results.
        """
        try:
            # Load test data
            test_data = self.load_data(str(self.config.test_data_path))
            
            # Load trained model
            model = self.load_model(str(self.config.model_path))
            
            # Prepare features and target
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]
            
            # Make predictions
            predicted_qualities = model.predict(test_x)
            y_prob = model.predict_proba(test_x)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate metrics
            metrics = self.eval_metrics(test_y.values.flatten(), predicted_qualities)
            
            # Create confusion matrix
            cm_artifact = self.create_confusion_matrix_artifact(
                test_y.values.flatten(), 
                predicted_qualities
            )
            
            # Convert numpy types to Python types
            cm_artifact = self._convert_numpy_types(cm_artifact)
            
            # Combine all metrics
            all_metrics = {
                **metrics,
                "confusion_matrix": cm_artifact["confusion_matrix"]
            }
            
            # Save metrics locally
            save_json(path=Path(self.config.metric_file_name), data=all_metrics)
            logger.info(f"Evaluation results saved to: {self.config.metric_file_name}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise e