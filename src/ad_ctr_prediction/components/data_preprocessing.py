import json
import numpy as np
import pandas as pd
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.entity.config_entity import DataPreprocessingConfig
from src.ad_ctr_prediction.utils.common import save_json
from pathlib import Path

class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):

        self.config = config

        self.data = pd.read_csv(
            self.config.input_data_path
        )

        self.report = {}

    def remove_duplicates(self):

        before_rows = len(self.data)

        self.data.drop_duplicates(inplace=True)

        after_rows = len(self.data)

        removed = (before_rows - after_rows)

        self.report["duplicates_removed"] = removed

        logger.info(
            f"Removed {removed} duplicates"
        )  

    def handle_infinite_values(self):

        numeric_cols = (self.data.select_dtypes(include=np.number).columns)

        total_inf = 0

        for col in numeric_cols:

            count = np.isinf(self.data[col]).sum()

            total_inf += count

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.report["infinite_values_replaced"] = int(total_inf)    

    def handle_missing_numerical(self):

        numeric_cols = (
            self.data.select_dtypes(include=np.number).columns)

        for col in numeric_cols:

            self.data[col] = (
                self.data[col].fillna(self.data[col].median())
            )

        logger.info(
            "Numerical missing values handled"
        )      

    def handle_missing_categorical(self):

        cat_cols = (
            self.data.select_dtypes(include=["object"]).columns
        )

        for col in cat_cols:

            self.data[col] = (
                self.data[col].fillna("Unknown")
            )

        logger.info(
            "Categorical missing values handled"
        )    

    def clean_invalid_ctr_values(self):

        ctr_columns = [

            "user_historical_ctr",

            "user_conversion_rate",

            "advertiser_historical_ctr",

            "advertiser_budget_utilization"
        ]

        before_rows = len(
            self.data
        )

        for col in ctr_columns:

            if col in self.data.columns:

                self.data = self.data[
                    (
                        self.data[col] >= 0
                    )
                    &
                    (
                        self.data[col] <= 1
                    )
                ]

        removed_rows = (before_rows - len(self.data))

        self.report["invalid_ctr_rows_removed"] = int(removed_rows)   

        logger.info(
            f"Removed {removed_rows} rows with invalid CTR values"
        ) 

    def process_timestamp(self):

        if ("timestamp" not in self.data.columns):
            return

        self.data["timestamp"] = (
            pd.to_datetime(
                self.data["timestamp"]
            )
        )

        self.data["year"] = (
            self.data["timestamp"].dt.year
        )

        logger.info(
            "Processed timestamp into year"
        )

    def handle_outliers(self):

        numeric_cols = (
            self.data.select_dtypes(include=np.number).columns)

        for col in numeric_cols:

            lower = (
                self.data[col].quantile(0.01)
            )

            upper = (
                self.data[col].quantile(0.99)
            )

            self.data[col] = (
                self.data[col].clip(lower, upper)
                )

        logger.info(
            "Outliers handled"
        )    
    
    def save_data(self):

        self.data.to_csv(
            self.config.output_data_path,
            index=False
        )

        save_json(Path(self.config.preprocessing_report_path), self.report)

        logger.info(
            "Preprocessed data saved"
        )

    def initiate_data_preprocessing(self):

        logger.info(
            "Starting preprocessing"
        )

        self.remove_duplicates()

        self.handle_infinite_values()

        self.handle_missing_numerical()

        self.handle_missing_categorical()

        self.clean_invalid_ctr_values()

        self.process_timestamp()

        self.handle_outliers()

        self.save_data()

        logger.info(
            "Preprocessing completed"
        )

        return (self.config.output_data_path) 