from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
 
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    REPORT_FILE: Path
    DRIFT_REPORT_FILE: Path
    STATS_REPORT_FILE: Path
    all_schema: dict
    numerical_ranges: dict
    categorical_values: dict
    thresholds: dict
    high_cardinality_columns: list
    leakage_columns: list
    timestamp_column: str
    target_column: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_data_path: Path
    output_data_path: Path
    preprocessing_report_path: Path    


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    input_data_path: Path
    output_data_path: Path
    feature_report_path: Path    


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    input_data_path: Path
    feature_selector_path: Path
    feature_names_path: Path
    metadata_path: Path
    split_artifacts_dir: Path
    train_file_path: Path
    validation_file_path: Path
    test_file_path: Path
    test_size: float
    validation_size: float
    random_state: int
    target_column: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_file_path: Path
    validation_file_path: Path
    model_file_path: Path
    metrics_file_path: Path
    model_params: dict
    target_column: str    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str    

@dataclass(frozen=True)
class ModelPromotionConfig:
    root_dir: Path
    metrics_file_path: Path
    model_file_path: Path
    production_model_path: Path
    registered_model_name: str
    target_stage: str
    mlflow_uri: str
    promote_metric: str
    promote_threshold: float
    archive_existing_versions: bool
    copy_local_model: bool     
