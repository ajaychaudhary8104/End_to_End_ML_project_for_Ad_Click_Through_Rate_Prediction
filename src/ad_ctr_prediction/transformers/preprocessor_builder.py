from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, RobustScaler, StandardScaler)
from src.ad_ctr_prediction.transformers.frequency_encoder import FrequencyEncoder


def create_numerical_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(strategy="median")
            ),

            (
                "scaler", RobustScaler()
            )
        ]
    )

def create_low_card_pipeline():

    return Pipeline(steps=[
            (
                "imputer", SimpleImputer(strategy="most_frequent")
            ),

            (
                "encoder", OneHotEncoder(handle_unknown="ignore",sparse_output= False)
            )
        ]
    )

def create_high_card_pipeline():

    return Pipeline(

        steps=[

            (
                "encoder", FrequencyEncoder()
            ),

            (
                "scaler", StandardScaler()
            )
        ]
    )

def build_preprocessor(numerical_columns, low_card_columns, high_card_columns):

    return ColumnTransformer(
        transformers=[
            (
                "num", create_numerical_pipeline(), numerical_columns
            ),

            (
                "low_card", create_low_card_pipeline(), low_card_columns
            ),

            (
                "high_card", create_high_card_pipeline(), high_card_columns
            )
        ]
    )

