import os
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from ad_ctr_prediction.entity.config_entity import DataValidationConfig
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.utils.common import save_json

warnings.filterwarnings("ignore")


class DataValidation:
    def __init__(self,config: DataValidationConfig):

        self.config = config

        self.data = pd.read_csv(
            os.path.join(
                self.config.unzip_data_dir,
                "CTR_raw_data.csv"
            )
        )

        self.validation_report = {}

        self.dataset_statistics = {}

        self.drift_report = {}

        logger.info(
            f"Dataset Loaded Successfully "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # HELPER METHODS
    # =====================================================

    def _update_report(self, validation_name: str, status: bool, details=None):

        self.validation_report[
            validation_name
        ] = {

            "status": status,

            "details": details
        }

    def _get_missing_percentage(self, series: pd.Series) -> float:

        return round(
            (
                series.isnull().sum()
                /
                len(series)) * 100, 4
        )

    def _save_reports(self):

        save_json(
            self.config.REPORT_FILE,
            self.validation_report
        )

        save_json(
            self.config.STATS_REPORT_FILE,
            self.dataset_statistics
        )

        if len(self.drift_report) > 0:

            save_json(
                self.config.DRIFT_REPORT_FILE,
                self.drift_report
            )

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    def generate_dataset_statistics(self):

        try:

            stats = {

                "rows":
                    int(
                        self.data.shape[0]
                    ),

                "columns":
                    int(
                        self.data.shape[1]
                    ),

                "duplicates":
                    int(
                        self.data.duplicated().sum()
                    ),

                "memory_usage_mb":
                    round(
                        self.data.memory_usage(deep=True).sum()
                        /
                        1024
                        /
                        1024,
                        2
                    )
            }

            if (self.config.target_column in self.data.columns):

                stats[
                    "target_distribution"
                ] = (
                    self.data[
                        self.config.target_column
                    ]
                    .value_counts(
                        normalize=True
                    )
                    .to_dict()
                )

            self.dataset_statistics = stats

            logger.info(
                "Dataset Statistics Generated"
            )

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # SCHEMA VALIDATION
    # =====================================================

    def validate_schema(self) -> bool:

        try:

            expected_columns = list(
                self.config.all_schema.keys()
            )

            actual_columns = list(
                self.data.columns
            )

            missing_columns = []

            extra_columns = []

            for col in expected_columns:

                if col not in actual_columns:

                    missing_columns.append(
                        col
                    )

            for col in actual_columns:

                if col not in expected_columns:

                    extra_columns.append(
                        col
                    )

            validation_passed = (
                len(missing_columns) == 0
            )

            self._update_report(
                validation_name=
                "Schema Validation",

                status=
                validation_passed,

                details={
                    "missing_columns":
                        missing_columns,

                    "extra_columns":
                        extra_columns
                }
            )

            if validation_passed:

                logger.info(
                    "✓ Schema Validation Passed"
                )

            else:

                logger.warning(
                    f"Missing Columns: "
                    f"{missing_columns}"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # DATATYPE VALIDATION
    # =====================================================

    def validate_datatypes(self) -> bool:

        try:

            schema = (
                self.config.all_schema
            )

            mismatches = []

            for col, expected_dtype in (
                schema.items()
            ):

                if col not in self.data.columns:
                    continue

                actual_dtype = str(
                    self.data[col].dtype
                )

                if (
                    actual_dtype
                    != expected_dtype
                ):

                    mismatches.append(
                        {
                            "column": col,
                            "expected":
                                expected_dtype,
                            "actual":
                                actual_dtype
                        }
                    )

            validation_passed = (
                len(mismatches) == 0
            )

            self._update_report(
                "Datatype Validation",
                validation_passed,
                mismatches
            )

            if validation_passed:

                logger.info(
                    "✓ Datatype Validation Passed"
                )

            else:

                logger.warning(
                    f"{len(mismatches)} "
                    f"Datatype Mismatches Found"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # MISSING VALUE VALIDATION
    # =====================================================

    def validate_missing_values(
        self
    ) -> bool:

        try:

            threshold = (
                self.config.thresholds.missing_value_threshold
            )

            problematic_columns = []

            for col in self.data.columns:

                missing_ratio = (
                    self.data[col]
                    .isnull()
                    .mean()
                )

                if (
                    missing_ratio
                    >
                    threshold
                ):

                    problematic_columns.append(
                        {
                            "column": col,

                            "missing_pct":
                                round(
                                    missing_ratio
                                    * 100,
                                    2
                                )
                        }
                    )

            validation_passed = (
                len(problematic_columns)
                == 0
            )

            self._update_report(
                "Missing Value Validation",
                validation_passed,
                problematic_columns
            )

            if validation_passed:

                logger.info(
                    "✓ Missing Value Validation Passed"
                )

            else:

                logger.warning(
                    f"{len(problematic_columns)} "
                    f"Columns Exceed Missing "
                    f"Threshold"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # DUPLICATE VALIDATION
    # =====================================================

    def validate_duplicates(
        self
    ) -> bool:

        try:

            duplicate_count = int(
                self.data
                .duplicated()
                .sum()
            )

            duplicate_pct = (
                duplicate_count
                /
                len(self.data)
            )

            max_allowed = (
                self.config.thresholds
                .max_duplicate_percentage
            )

            validation_passed = (
                duplicate_pct
                <=
                max_allowed
            )

            self._update_report(
                "Duplicate Validation",
                validation_passed,
                {
                    "duplicates":
                        duplicate_count,

                    "duplicate_percentage":
                        round(
                            duplicate_pct
                            * 100,
                            4
                        )
                }
            )

            if validation_passed:

                logger.info(
                    "✓ Duplicate Validation Passed"
                )

            else:

                logger.warning(
                    f"Duplicate Percentage "
                    f"Exceeded Threshold"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # NUMERICAL RANGE VALIDATION
    # =====================================================

    def validate_numerical_ranges(
        self
    ) -> bool:

        try:

            validation_issues = []

            ranges = (
                self.config
                .numerical_ranges
            )

            for col, limits in (
                ranges.items()
            ):

                if col not in self.data.columns:
                    continue

                min_value = limits.min
                max_value = limits.max

                invalid_rows = self.data[
                    (
                        self.data[col]
                        < min_value
                    )
                    |
                    (
                        self.data[col]
                        > max_value
                    )
                ]

                if len(
                    invalid_rows
                ) > 0:

                    validation_issues.append(
                        {
                            "column": col,

                            "invalid_rows":
                                len(
                                    invalid_rows
                                ),

                            "expected_range":
                                f"{min_value}"
                                f" - "
                                f"{max_value}"
                        }
                    )

            validation_passed = (
                len(validation_issues)
                == 0
            )

            self._update_report(
                "Numerical Range Validation",
                validation_passed,
                validation_issues
            )

            if validation_passed:

                logger.info(
                    "✓ Numerical Range "
                    "Validation Passed"
                )

            else:

                logger.warning(
                    f"{len(validation_issues)} "
                    f"Range Violations Found"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # TARGET VALIDATION
    # =====================================================

    def validate_target(self) -> bool:

        try:

            target_col = (
                self.config.target_column
            )

            if target_col not in self.data.columns:

                self._update_report(
                    "Target Validation",
                    False,
                    "Target column not found"
                )

                return False

            invalid_values = self.data[
                ~self.data[target_col]
                .isin([0, 1])
            ]

            validation_passed = (
                len(invalid_values) == 0
            )

            self._update_report(
                "Target Validation",
                validation_passed,
                {
                    "invalid_records":
                        len(invalid_values)
                }
            )

            if validation_passed:

                logger.info(
                    "✓ Target Validation Passed"
                )

            else:

                logger.warning(
                    "Invalid target values detected"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # CATEGORICAL VALIDATION
    # =====================================================

    def validate_categorical_values(
        self
    ) -> bool:

        try:

            validation_issues = []

            allowed_categories = (
                self.config
                .categorical_values
            )

            for col, allowed in (
                allowed_categories.items()
            ):

                if col not in self.data.columns:
                    continue

                invalid_rows = self.data[
                    ~self.data[col]
                    .isin(list(allowed))
                ]

                if len(invalid_rows) > 0:

                    validation_issues.append(
                        {
                            "column": col,

                            "invalid_count":
                                len(
                                    invalid_rows
                                ),

                            "allowed_values":
                                list(
                                    allowed
                                )
                        }
                    )

            validation_passed = (
                len(validation_issues)
                == 0
            )

            self._update_report(
                "Categorical Validation",
                validation_passed,
                validation_issues
            )

            if validation_passed:

                logger.info(
                    "✓ Categorical Validation Passed"
                )

            else:

                logger.warning(
                    "Categorical Validation Failed"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # TIMESTAMP VALIDATION
    # =====================================================

    def validate_timestamp(self) -> bool:

        try:

            timestamp_col = (
                self.config.timestamp_column
            )

            if timestamp_col not in self.data.columns:

                return True

            timestamps = pd.to_datetime(
                self.data[timestamp_col],
                errors="coerce"
            )

            invalid_dates = int(
                timestamps.isnull().sum()
            )

            future_dates = int(
                (
                    timestamps
                    >
                    pd.Timestamp.now()
                ).sum()
            )

            validation_passed = (
                invalid_dates == 0
                and future_dates == 0
            )

            self._update_report(
                "Timestamp Validation",
                validation_passed,
                {
                    "invalid_dates":
                        invalid_dates,

                    "future_dates":
                        future_dates
                }
            )

            if validation_passed:

                logger.info(
                    "✓ Timestamp Validation Passed"
                )

            else:

                logger.warning(
                    "Timestamp Validation Failed"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # BUSINESS RULE VALIDATION
    # =====================================================

    def validate_business_rules(
        self
    ) -> bool:

        try:

            issues = []

            if (
                "user_total_clicks"
                in self.data.columns
                and
                "user_total_impressions"
                in self.data.columns
            ):

                invalid_clicks = self.data[
                    self.data[
                        "user_total_clicks"
                    ]
                    >
                    self.data[
                        "user_total_impressions"
                    ]
                ]

                if len(
                    invalid_clicks
                ) > 0:

                    issues.append(
                        {
                            "rule":
                                "Clicks <= Impressions",

                            "violations":
                                len(
                                    invalid_clicks
                                )
                        }
                    )

            if (
                "user_historical_ctr"
                in self.data.columns
            ):

                calculated_ctr = (
                    self.data[
                        "user_total_clicks"
                    ]
                    /
                    self.data[
                        "user_total_impressions"
                    ].replace(
                        0,
                        np.nan
                    )
                )

                mismatch = (
                    np.abs(
                        calculated_ctr
                        -
                        self.data[
                            "user_historical_ctr"
                        ]
                    )
                    >
                    0.01
                )

                mismatch_count = int(
                    mismatch.sum()
                )

                if mismatch_count > 0:

                    issues.append(
                        {
                            "rule":
                                "CTR Consistency",

                            "violations":
                                mismatch_count
                        }
                    )

            validation_passed = (
                len(issues) == 0
            )

            self._update_report(
                "Business Rule Validation",
                validation_passed,
                issues
            )

            if validation_passed:

                logger.info(
                    "✓ Business Rule Validation Passed"
                )

            else:

                logger.warning(
                    "Business Rule Violations Found"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # INFINITE VALUE VALIDATION
    # =====================================================

    def validate_infinite_values(
        self
    ) -> bool:

        try:

            numeric_cols = (
                self.data
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            problematic_columns = []

            for col in numeric_cols:

                inf_count = int(
                    np.isinf(
                        self.data[col]
                    ).sum()
                )

                if inf_count > 0:

                    problematic_columns.append(
                        {
                            "column": col,
                            "count": inf_count
                        }
                    )

            validation_passed = (
                len(problematic_columns)
                == 0
            )

            self._update_report(
                "Infinite Value Validation",
                validation_passed,
                problematic_columns
            )

            if validation_passed:

                logger.info(
                    "✓ Infinite Value Validation Passed"
                )

            else:

                logger.warning(
                    "Infinite Values Found"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # OUTLIER VALIDATION
    # =====================================================

    def validate_outliers(self) -> bool:

        try:

            outlier_report = {}

            numeric_cols = (
                self.data
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            for col in numeric_cols:

                q1 = self.data[col].quantile(
                    0.25
                )

                q3 = self.data[col].quantile(
                    0.75
                )

                iqr = q3 - q1

                lower = (
                    q1
                    - 1.5 * iqr
                )

                upper = (
                    q3
                    + 1.5 * iqr
                )

                outliers = (
                    (
                        self.data[col]
                        < lower
                    )
                    |
                    (
                        self.data[col]
                        > upper
                    )
                ).sum()

                outlier_report[col] = int(
                    outliers
                )

            self._update_report(
                "Outlier Validation",
                True,
                outlier_report
            )

            logger.info(
                "✓ Outlier Analysis Completed"
            )

            return True

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # CARDINALITY VALIDATION
    # =====================================================

    def validate_cardinality(self) -> bool:

        try:

            report = {}

            for col in (
                self.config
                .high_cardinality_columns
            ):

                if col not in self.data.columns:
                    continue

                report[col] = int(
                    self.data[col]
                    .nunique()
                )

            self._update_report(
                "Cardinality Validation",
                True,
                report
            )

            logger.info(
                "✓ Cardinality Analysis Completed"
            )

            return True

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # LEAKAGE VALIDATION
    # =====================================================

    def validate_data_leakage(
        self
    ) -> bool:

        try:

            leakage_columns = []

            dataset_columns = [
                col.lower()
                for col in self.data.columns
            ]

            for leak_col in (
                self.config
                .leakage_columns
            ):

                if (
                    leak_col.lower()
                    in dataset_columns
                ):

                    leakage_columns.append(
                        leak_col
                    )

            validation_passed = (
                len(leakage_columns)
                == 0
            )

            self._update_report(
                "Leakage Validation",
                validation_passed,
                leakage_columns
            )

            if validation_passed:

                logger.info(
                    "✓ Leakage Validation Passed"
                )

            else:

                logger.warning(
                    f"Potential Leakage Columns: "
                    f"{leakage_columns}"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # CORRELATION VALIDATION
    # =====================================================

    def validate_correlation(self) -> bool:

        try:

            threshold = (
                self.config.thresholds
                .correlation_threshold
            )

            numeric_df = (
                self.data
                .select_dtypes(
                    include=np.number
                )
            )

            if numeric_df.shape[1] < 2:

                self._update_report(
                    "Correlation Validation",
                    True,
                    {}
                )

                return True

            correlation_matrix = (
                numeric_df
                .corr()
                .abs()
            )

            upper_triangle = (
                correlation_matrix.where(
                    np.triu(
                        np.ones(
                            correlation_matrix.shape
                        ),
                        k=1
                    ).astype(bool)
                )
            )

            high_corr_pairs = []

            for column in upper_triangle.columns:

                correlated_features = (
                    upper_triangle.index[
                        upper_triangle[column]
                        >
                        threshold
                    ]
                    .tolist()
                )

                for feature in correlated_features:

                    high_corr_pairs.append(
                        {
                            "feature_1":
                                feature,

                            "feature_2":
                                column,

                            "correlation":
                                round(
                                    float(
                                        upper_triangle
                                        .loc[
                                            feature,
                                            column
                                        ]
                                    ),
                                    4
                                )
                        }
                    )

            self._update_report(
                "Correlation Validation",
                True,
                high_corr_pairs
            )

            logger.info(
                "✓ Correlation Analysis Completed"
            )

            return True

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # CLASS DISTRIBUTION VALIDATION
    # =====================================================

    def validate_class_distribution(
        self
    ) -> bool:

        try:

            target_col = (
                self.config.target_column
            )

            if target_col not in self.data.columns:

                return False

            distribution = (
                self.data[target_col]
                .value_counts(
                    normalize=True
                )
                .to_dict()
            )

            minimum_class = min(
                distribution.values()
            )

            threshold = (
                self.config.thresholds
                .minimum_minority_class
            )

            validation_passed = (
                minimum_class
                >=
                threshold
            )

            self._update_report(
                "Class Distribution Validation",
                validation_passed,
                distribution
            )

            if validation_passed:

                logger.info(
                    "✓ Class Distribution Validation Passed"
                )

            else:

                logger.warning(
                    "Severe Class Imbalance Detected"
                )

            return validation_passed

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # DATA DRIFT VALIDATION
    # =====================================================

    def validate_data_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> bool:

        try:

            threshold = (
                self.config.thresholds
                .drift_pvalue_threshold
            )

            numerical_columns = (
                train_df
                .select_dtypes(
                    include=np.number
                )
                .columns
            )

            overall_pass = True

            drift_report = {}

            for col in numerical_columns:

                statistic, p_value = (
                    ks_2samp(
                        train_df[col]
                        .dropna(),

                        test_df[col]
                        .dropna()
                    )
                )

                drift_detected = (
                    p_value
                    <
                    threshold
                )

                if drift_detected:

                    overall_pass = False

                drift_report[col] = {

                    "ks_statistic":
                        round(
                            float(
                                statistic
                            ),
                            4
                        ),

                    "p_value":
                        round(
                            float(
                                p_value
                            ),
                            6
                        ),

                    "drift_detected":
                        drift_detected
                }

            self.drift_report = (
                drift_report
            )

            self._update_report(
                "Data Drift Validation",
                overall_pass,
                drift_report
            )

            logger.info(
                "✓ Drift Analysis Completed"
            )

            return overall_pass

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # SAVE STATUS FILE
    # =====================================================

    def save_validation_status(
        self,
        validation_results: dict
    ):

        try:

            os.makedirs(
                self.config.root_dir,
                exist_ok=True
            )

            overall_status = all(
                validation_results.values()
            )

            with open(
                self.config.STATUS_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "CTR DATA VALIDATION REPORT\n"
                )

                file.write(
                    "=" * 60
                )

                file.write(
                    "\n\n"
                )

                for (
                    validation_name,
                    status
                ) in validation_results.items():

                    file.write(
                        f"{validation_name}: "
                    )

                    file.write(
                        f"{'PASSED' if status else 'FAILED'}\n"
                    )

                file.write(
                    "\n"
                )

                file.write(
                    "=" * 60
                )

                file.write(
                    "\n"
                )

                file.write(
                    f"OVERALL STATUS: "
                )

                file.write(
                    f"{'PASSED' if overall_status else 'FAILED'}"
                )

            logger.info(
                "Validation Status Saved"
            )

        except Exception as e:

            logger.error(str(e))

            raise e

    # =====================================================
    # MAIN VALIDATION PIPELINE
    # =====================================================

    def initiate_data_validation(self) -> bool:

        try:

            logger.info(
                "Starting Data Validation"
            )

            self.generate_dataset_statistics()

            validation_results = {

                "Schema Validation":
                    self.validate_schema(),

                "Datatype Validation":
                    self.validate_datatypes(),

                "Missing Value Validation":
                    self.validate_missing_values(),

                "Duplicate Validation":
                    self.validate_duplicates(),

                "Numerical Range Validation":
                    self.validate_numerical_ranges(),

                "Target Validation":
                    self.validate_target(),

                "Categorical Validation":
                    self.validate_categorical_values(),

                "Timestamp Validation":
                    self.validate_timestamp(),

                "Business Rule Validation":
                    self.validate_business_rules(),

                "Infinite Value Validation":
                    self.validate_infinite_values(),

                "Outlier Validation":
                    self.validate_outliers(),

                "Cardinality Validation":
                    self.validate_cardinality(),

                "Leakage Validation":
                    self.validate_data_leakage(),

                "Correlation Validation":
                    self.validate_correlation(),

                "Class Distribution Validation":
                    self.validate_class_distribution()
            }

            self.save_validation_status(
                validation_results
            )

            self._save_reports()

            overall_status = all(
                validation_results.values()
            )

            if overall_status:

                logger.info(
                    "✓ ALL VALIDATIONS PASSED"
                )

            else:

                logger.warning(
                    "✗ SOME VALIDATIONS FAILED"
                )

            return overall_status

        except Exception as e:

            logger.error(str(e))

            raise e