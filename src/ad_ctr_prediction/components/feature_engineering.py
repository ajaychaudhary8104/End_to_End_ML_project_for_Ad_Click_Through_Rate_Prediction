import numpy as np
import pandas as pd
from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.utils.common import save_json
from src.ad_ctr_prediction.entity.config_entity import FeatureEngineeringConfig
from pathlib import Path


class FeatureEngineering:

    def __init__(self, config: FeatureEngineeringConfig):

        self.config = config

        self.data = pd.read_csv(
            self.config.input_data_path
        )

        self.feature_report = {}

    def create_user_click_ratio(self):

        self.data["user_click_ratio"] = (
            self.data["user_total_clicks"] / (self.data["user_total_impressions"] + 1)
        )

        logger.info(
            "user_click_ratio created"
        )    

    def create_activity_score(self):

        self.data["activity_score"] = (

            self.data["user_days_active"] * self.data["user_avg_session_duration"]
        )

        logger.info(
            "activity_score created"
        )    

    def create_budget_efficiency(self):

        self.data["budget_efficiency"] = (

            self.data["advertiser_historical_ctr"]

            *

            self.data["advertiser_budget_utilization"]
        )    

        logger.info("budget_efficiency created")

    def create_ad_strength(self):

        self.data["ad_strength"] = (

            self.data["ad_quality_score"]

            *

            self.data["advertiser_historical_ctr"]
        )    

        logger.info("ad_strength created")

    def create_engagement_score(self):

        self.data["engagement_score"] = (

            self.data["page_quality_score"]

            *

            self.data["page_dwell_time"]
        )    

        logger.info("engagement_score created")

    def create_peak_hour_feature(self):

        peak_hours = [7,8,9,10,17,18,19,20,21,22]

        self.data["peak_hour"] = (

            self.data["hour_of_day"].isin(peak_hours).astype(int)
        )   

        logger.info("peak_hour feature created")

    def create_night_user_feature(self):

        self.data["night_user"] = (

            (
                self.data["hour_of_day"]
                >= 22
            )

            |

            (
                self.data["hour_of_day"]
                <= 5
            )
            ).astype(int)     

        logger.info("night_user feature created")

    def create_weekend_evening(self):

        self.data["weekend_evening"] = (

            (
                self.data["is_weekend"] == 1
            )

            &

            (
                self.data["hour_of_day"] >= 18
            )

        ).astype(int)    

        logger.info("weekend_evening feature created")

    def create_user_ad_affinity(self):

        self.data["user_ad_affinity"] = (

            self.data["user_historical_ctr"]

            *

            self.data["ad_quality_score"]
        )    

        logger.info("user_ad_affinity feature created")

    def create_interest_quality(self):

        self.data["interest_quality"] = (

            self.data["interest_match"]

            *

            self.data["ad_quality_score"]
        )
        
        logger.info("interest_quality feature created")

    def create_user_page_affinity(self):

        self.data["user_page_affinity"] = (

            self.data["user_historical_ctr"]

            *

            self.data["page_quality_score"]
        )  

        logger.info("user_page_affinity feature created")

    def create_recent_clicker(self):

        self.data["recent_clicker"] = (

            self.data["time_since_last_click"] < 3600).astype(int)

        logger.info("recent_clicker feature created")

    def create_recent_impression(self):

        self.data["recent_impression"] = (

            self.data["time_since_last_impression"] < 1800).astype(int)    

        logger.info("recent_impression feature created")

    def create_ad_fatigue(self):

        self.data["ad_fatigue"] = (

            self.data["sequence_position"] > 5).astype(int)    

        logger.info("ad_fatigue feature created")

    def create_click_momentum(self):

        self.data["click_momentum"] = (

            self.data["previous_ad_clicks_today"]

            /

            (
                self.data["sequence_position"] + 1
            )
        )    

        logger.info("click_momentum feature created")

    def create_quarter_feature(self):

        self.data["quarter"] = (

            (
                self.data["month"] - 1
            ) // 3 + 1
        )    

        logger.info("quarter feature created")

    def create_month_cyclic(self):

        self.data["month_sin"] = np.sin(2 * np.pi * self.data["month"] / 12)

        self.data["month_cos"] = np.cos(2* np.pi * self.data["month"] / 12)    

        logger.info("month cyclic features created")

    def save_features(self):

        self.data.to_csv(self.config.output_data_path, index=False)

        self.feature_report["final_columns"] = len(self.data.columns)

        save_json(Path(self.config.feature_report_path), self.feature_report)  

        logger.info(
            f"Features saved to {self.config.output_data_path} and feature report saved to {self.config.feature_report_path}"
        )

    def initiate_feature_engineering(self):

        self.create_user_click_ratio()

        self.create_activity_score()

        self.create_budget_efficiency()

        self.create_ad_strength()

        self.create_engagement_score()

        self.create_peak_hour_feature()

        self.create_night_user_feature()

        self.create_weekend_evening()

        self.create_user_ad_affinity()

        self.create_interest_quality()

        self.create_user_page_affinity()

        self.create_recent_clicker()

        self.create_recent_impression()

        self.create_ad_fatigue()

        self.create_click_momentum()

        self.create_quarter_feature()

        self.create_month_cyclic()

        self.save_features()

        return (
            self.config.output_data_path
        )    