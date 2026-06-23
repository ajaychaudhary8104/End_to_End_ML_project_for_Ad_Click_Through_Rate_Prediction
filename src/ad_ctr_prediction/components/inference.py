from pathlib import Path
import joblib
import pandas as pd
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.entity.config_entity import ModelInferenceConfig
from src.ad_ctr_prediction.utils.common import create_directories
from typing import List, Dict, Any
from ad_ctr_prediction.transformers.ctr_feature_engineer import CTRFeatureEngineer


class ModelInference:
    """
    Production-grade inference pipeline using:
    - Feature Engineering
    - Saved preprocessor
    - Saved Feature Selector
    - Saved trained model

    
    """

    def __init__(self, config: ModelInferenceConfig):

        self.config = config

        create_directories([self.config.root_dir])

        self.preprocessor = self.load_preprocessor()

        self.model = self.load_model()

        self.feature_selector = self.load_feature_selector()

        self.feature_names = self.load_feature_names()

    # =========================================================
    # LOAD PREPROCESSOR
    # =========================================================
    def load_preprocessor(self):

        preprocessor_path = Path(self.config.preprocessor_path)

        if not preprocessor_path.exists():

            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        logger.info(
            f"Loading preprocessor from: {preprocessor_path}"
        )

        preprocessor = joblib.load(preprocessor_path)

        logger.info("Preprocessor loaded successfully")

        return preprocessor
    
    # =========================================================
    # LOAD FEATURE SELECTOR
    # =========================================================
    def load_feature_selector(self):

        feature_selector_path = Path(self.config.feature_selector_path)

        if not feature_selector_path.exists():

            raise FileNotFoundError(
                f"Feature selector not found: {feature_selector_path}"
            )

        logger.info(
            f"Loading feature selector from: {feature_selector_path}"
        )

        feature_selector = joblib.load(feature_selector_path)

        logger.info("Feature selector loaded successfully")

        return feature_selector
    
    # =========================================================
    # LOAD FEATURE NAMES
    # =========================================================
    def load_feature_names(self) -> List[str]:

        feature_names_path = Path(self.config.feature_names_path)

        if not feature_names_path.exists():

            raise FileNotFoundError(
                f"Feature names file not found: {feature_names_path}"
            )

        logger.info(
            f"Loading feature names from: {feature_names_path}"
        )

        feature_names = joblib.load(feature_names_path)

        logger.info("Feature names loaded successfully")

        return feature_names

    # =========================================================
    # LOAD MODEL
    # =========================================================
    def load_model(self):

        model_path = Path(self.config.model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        logger.info(
            f"Loading model from: {model_path}"
        )

        model = joblib.load(model_path)

        logger.info("Model loaded successfully")

        return model

    # =========================================================
    # LOAD INPUT DATA
    # =========================================================
    def load_input_data(self) -> pd.DataFrame:

        input_path = Path(
            self.config.input_data_path
        )

        if not input_path.exists():


            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        logger.info(
            f"Loading input data from: {input_path}"
        )

        return pd.read_csv(input_path)
    
    # ======================================================
    # Process Timestamp
    # ======================================================
    def process_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:

        if ("timestamp" not in data.columns):
            return data


        data["timestamp"] = (
            pd.to_datetime(
                data["timestamp"]
            )
        )

        data["year"] = (
            data["timestamp"].dt.year
        )

        logger.info(
            "Processed timestamp into year"
        )
        return data


    # ======================================================
    # Drop Columns
    # ======================================================
    def drop_columns(self, data):

        drop_cols = [
            col for col in self.config.drop_columns
            if col in data.columns
        ]

        if drop_cols:

            logger.info(
                f"Dropping columns: {drop_cols}"
            )

            data = data.drop(
                columns=drop_cols
            )

        return data


    # =========================================================
    # CREATE DERIVED FEATURES
    # =========================================================
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Creating derived features")
        feature_engineer = CTRFeatureEngineer()
        data = feature_engineer.transform(data)
        return data



    # =========================================================
    # PREPARE FEATURES
    # =========================================================
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:

        if (hasattr(self.config, "target_column") and self.config.target_column in data.columns):

            logger.info(
                f"Dropping target column: {self.config.target_column}"
            )

            data = data.drop(
                columns=[self.config.target_column]
            )

        return data

    # =========================================================
    # PREPROCESS DATA
    # =========================================================
    def preprocess_data(self, data: pd.DataFrame):

        logger.info("Starting preprocessing")

        transformed_data = (
            self.preprocessor.transform(data)
        )
        try:

            feature_names = self.preprocessor.get_feature_names_out()

        except Exception as e:

            logger.warning(
                f"Could not extract feature names: {e}"
            )

            feature_names = None

        transformed_data = pd.DataFrame(
            transformed_data,
            columns=feature_names
        )

        logger.info("Preprocessing completed")


        return transformed_data

    # =========================================================
    # PREDICT
    # =========================================================
    def predict(self,input_df: pd.DataFrame) -> pd.DataFrame:

        if input_df.empty:

            raise ValueError("Input dataframe is empty")

        logger.info(f"Running inference on {len(input_df)} records")
        # ======================================================
        # Process Timestamp
        # ======================================================
        input_df = self.process_timestamp(input_df.copy())

        # -----------------------------------------
        # Drop Insignificant Columns
        #------------------------------------------
        input_df = self.drop_columns(input_df.copy())

        # -----------------------------------------
        # Remove target column if present
        # -----------------------------------------
        features = self.prepare_features(input_df)
        
        # -----------------------------------------
        # Create derived features
        # -----------------------------------------
        features = self.create_derived_features(features)

        # -----------------------------------------
        # Transform using saved preprocessor
        # -----------------------------------------
        transformed_features = (self.preprocess_data(features))
        logger.info(f"shape of transformed_features is {transformed_features.shape}")

        # -----------------------------------------
        # Select Features
        # -----------------------------------------
        transformed_features = self.feature_selector.transform(transformed_features)
        transformed_features = pd.DataFrame(transformed_features)

        # -----------------------------------------
        # Predict
        # -----------------------------------------
        predictions = self.model.predict(transformed_features)

        results = pd.DataFrame({"prediction": predictions})

        # =========================================
        # PROBABILITY PREDICTIONS
        # =========================================
        if hasattr(self.model, "predict_proba"):

            try:

                probabilities = (self.model.predict_proba(transformed_features)[:, 1])

                results["prediction_probability"] = probabilities

            except Exception as exc:

                logger.warning(f"Probability prediction failed: {exc}")

        return results

    # =========================================================
    # SAVE PREDICTIONS
    # =========================================================
    def save_predictions(self,input_df: pd.DataFrame,results: pd.DataFrame) -> Path:

        output_path = Path(self.config.prediction_output_path)

        output_path.parent.mkdir(parents=True,exist_ok=True)

        final_df = pd.concat([input_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

        final_df.to_csv(output_path,index=False)

        logger.info(f"Predictions saved to: {output_path}")

        return output_path

    # =========================================================
    # BATCH INFERENCE
    # =========================================================
    def run_batch_inference(self) -> Path:

        logger.info("Starting batch inference")

        input_df = self.load_input_data()

        results = self.predict(input_df)

        output_path = self.save_predictions(input_df=input_df,results=results)

        logger.info("Batch inference completed")

        return output_path

    # =========================================================
    # API PREDICTION
    # =========================================================
    def predict_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:

        if not records:
            raise ValueError("No records provided")

        input_df = pd.DataFrame(records)

        return self.predict(input_df)
