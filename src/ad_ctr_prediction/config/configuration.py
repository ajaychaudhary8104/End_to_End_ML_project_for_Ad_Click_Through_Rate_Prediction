from src.ad_ctr_prediction.constants import *
from src.ad_ctr_prediction.utils.common import read_yaml, create_directories
from src.ad_ctr_prediction.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig,
                                                         DataValidationConfig, FeatureEngineeringConfig ,
                                                           DataTransformationConfig, ModelTrainingConfig,
                                                           ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

     
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        schema = self.schema

        create_directories(
            [config.root_dir]
        )

        validation_config = (DataValidationConfig(
                root_dir=Path(config.root_dir),

                unzip_data_dir=Path(config.unzip_data_dir),

                STATUS_FILE=Path(config.STATUS_FILE),

                REPORT_FILE=Path(config.REPORT_FILE),

                DRIFT_REPORT_FILE=Path(config.DRIFT_REPORT_FILE),

                STATS_REPORT_FILE=Path(config.STATS_REPORT_FILE),

                all_schema=schema.COLUMNS,

                numerical_ranges=schema.NUMERICAL_RANGES,

                categorical_values=schema.CATEGORICAL_VALUES,

                thresholds=schema.THRESHOLDS,

                high_cardinality_columns=
                schema.HIGH_CARDINALITY_COLUMNS,

                leakage_columns=
                schema.LEAKAGE_COLUMNS,

                timestamp_column=
                schema.TIMESTAMP_COLUMN,

                target_column=
                schema.TARGET_COLUMN
            )
        )

        return validation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        create_directories(
            [config.root_dir]
        )

        data_preprocessing = DataPreprocessingConfig(

            root_dir=Path(config.root_dir),

            input_data_path=Path(config.input_data_path),

            output_data_path=Path(config.output_data_path),

            preprocessing_report_path=Path(config.preprocessing_report_path)
        )  

        return data_preprocessing
    
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config.feature_engineering

        create_directories(
            [config.root_dir]
        )

        feature_engineering_config = FeatureEngineeringConfig(

            root_dir=Path(config.root_dir),

            input_data_path=Path(config.input_data_path),

            output_data_path=Path(config.output_data_path),

            feature_report_path=Path(config.feature_report_path)
        )    

        return feature_engineering_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        create_directories(
            [config.root_dir]
        )

        data_transformation_config = DataTransformationConfig(root_dir=Path(config.root_dir),
                                                            input_data_path=Path(config.input_data_path),
                                                            feature_selector_path=Path(config.feature_selector_path),
                                                            feature_names_path=Path(config.feature_names_path),
                                                            metadata_path=Path(config.metadata_path),
                                                            split_artifacts_dir=Path(config.split_artifacts_dir),
                                                            train_file_path=Path(config.train_file_path),
                                                            validation_file_path=Path(config.validation_file_path),
                                                            test_file_path=Path(config.test_file_path),
                                                            test_size=float(config.test_size),
                                                            validation_size=float(config.validation_size),
                                                            random_state=int(config.random_state),
                                                            target_column=str(config.target_column)
                                                            )
        return data_transformation_config
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_file_path=config.train_file_path,
            validation_file_path=config.validation_file_path,
            model_file_path=config.model_file_path,
            metrics_file_path=config.metrics_file_path,
            model_params=dict(self.params.model_params),
            target_column=config.target_column
        )

        return model_training_config 
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params= dict(params.model_params),
            metric_file_name = config.metric_file_name,
            target_column = config.target_column,
            mlflow_uri= config.mlflow_uri
        )

        return model_evaluation_config