import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import (train_test_split)
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.transformers.feature_manager import (FeatureManager)
from src.ad_ctr_prediction.transformers.feature_selector import (CTRFeatureSelector)
from src.ad_ctr_prediction.transformers.preprocessor_builder import *
from src.ad_ctr_prediction.transformers.imbalance_handler import *
import pickle
from src.ad_ctr_prediction.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config: DataTransformationConfig):

        self.config = config
        self.data = pd.read_csv(config.input_data_path)
        self.preprocessor = None

    def split_data(self, data):

        target_col = (
            self.config.target_column
        )

        X = data.drop(
            columns=[target_col]
        )

        y = data[target_col]

        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        )

        relative_val_size = (
            self.config.validation_size /
            (1 - self.config.test_size)
        )

        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=relative_val_size,
                random_state=self.config.random_state,
                stratify=y_train_val
            )
        )

        logger.info(
            f"Train shape: {X_train.shape}"
        )

        logger.info(
            f"Validation shape: {X_val.shape}"
        )

        logger.info(
            f"Test shape: {X_test.shape}"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )    
    
    def transform_data(self, X_train, X_val, X_test):
        
        # Numeric Columns
        numerical_columns = [col for col in X_train.columns if col not in FeatureManager.HIGH_CARDINALITY_COLUMNS and col not in 
                             FeatureManager.LOW_CARDINALITY_COLUMNS and X_train[col].dtype != "object"] 
            

        self.preprocessor = build_preprocessor(numerical_columns, 
                                               FeatureManager.LOW_CARDINALITY_COLUMNS, FeatureManager.HIGH_CARDINALITY_COLUMNS)

        logger.info(
            "Fitting preprocessor on train data"
        )

        X_train = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        X_val = (
            self.preprocessor.transform(
                X_val
            )
        )

        X_test = (
            self.preprocessor.transform(
                X_test
            )
        )
        
        try:

            feature_names = self.preprocessor.get_feature_names_out()
            

            joblib.dump(feature_names, self.config.feature_names_path)

        except Exception as e:

            logger.warning(
                f"Could not extract feature names: {e}"
            )

            feature_names = None

        X_train = pd.DataFrame(
            X_train,
            columns=feature_names
        )

        X_val = pd.DataFrame(
            X_val,
            columns=feature_names
        )

        X_test = pd.DataFrame(
            X_test,
            columns=feature_names
        )

        return X_train, X_val, X_test
    
    def save_preprocessor(self):

        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        preprocessor_path = os.path.join(
            self.config.root_dir,
            "preprocessor.pkl"
        )

        with open(preprocessor_path, "wb") as file:

            pickle.dump(self.preprocessor,file)

        logger.info(
            f"Preprocessor saved to: "
            f"{preprocessor_path}"
        )

    def save_data(self, X_train, X_val, X_test, y_train, y_val, y_test):

        os.makedirs(
            self.config.split_artifacts_dir,
            exist_ok=True
        )

        train_df = X_train.copy()
        train_df[self.config.target_column] = y_train.values

        val_df = X_val.copy()
        val_df[self.config.target_column] = y_val.values

        test_df = X_test.copy()
        test_df[self.config.target_column] = y_test.values

        train_df.to_csv(
            self.config.train_file_path,
            index=False
        )

        val_df.to_csv(
            self.config.validation_file_path,
            index=False
        )

        test_df.to_csv(
            self.config.test_file_path,
            index=False
        )

        logger.info(
            "Train/Validation/Test saved"
        )    


    def initiate_data_transformation(self):

        logger.info(
            "Starting Data Transformation"
        )

        df = self.data.copy()

        df.drop(columns= FeatureManager.IDENTIFIER_COLUMNS, inplace=True, errors="ignore")

        (X_train, X_val, X_test, y_train, y_val, y_test) = self.split_data(df)
        
        X_train, X_val, X_test = (
                self.transform_data(
                    X_train,
                    X_val,
                    X_test
                )
            )


        selector = CTRFeatureSelector(k=200)
        X_train = selector.fit_transform(X_train,y_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)

        X_train = pd.DataFrame(X_train)
        X_val = pd.DataFrame(X_val)
        X_test = pd.DataFrame(X_test)

        scale_pos_weight = get_scale_pos_weight(y_train)
        
        self.save_data(X_train,X_val,X_test,y_train,y_val,y_test)

        self.save_preprocessor()
        
        joblib.dump(selector, self.config.feature_selector_path)
        
        numerical_columns = [col for col in X_train.columns if col not in FeatureManager.HIGH_CARDINALITY_COLUMNS and col not in 
                             FeatureManager.LOW_CARDINALITY_COLUMNS and X_train[col].dtype != "object"] 
        
        metadata = {"num_features":
                    len(
                        numerical_columns
                    ),

                    "selected_features":
                        X_train.shape[1],

                    "scale_pos_weight":
                        float(
                            scale_pos_weight
                        ),

                    "train_rows":
                        int(
                            X_train.shape[0]
                        ),
                        
                    "validation_rows":
                        int(
                            X_val.shape[0]
                        ),
                    "test_rows":
                        int(
                            X_test.shape[0]
                        )
                }

        with open(self.config.metadata_path, "w") as file:
            json.dump(metadata, file,indent=4)
        
            logger.info("Transformation Completed")