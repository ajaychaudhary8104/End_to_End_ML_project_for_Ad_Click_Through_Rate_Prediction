import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CTRFeatureEngineer(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # =====================================
        # User Features
        # =====================================

        X["user_click_ratio"] = (
            X["user_total_clicks"]
            /
            (X["user_total_impressions"] + 1)
        )

        X["activity_score"] = (
            X["user_days_active"]
            *
            X["user_avg_session_duration"]
        )

        # =====================================
        # Advertiser Features
        # =====================================

        X["budget_efficiency"] = (
            X["advertiser_historical_ctr"]
            *
            X["advertiser_budget_utilization"]
        )

        X["ad_strength"] = (
            X["ad_quality_score"]
            *
            X["advertiser_historical_ctr"]
        )

        # =====================================
        # Page Features
        # =====================================

        X["engagement_score"] = (
            X["page_quality_score"]
            *
            X["page_dwell_time"]
        )

        # =====================================
        # Time Features
        # =====================================

        peak_hours = [
            7,8,9,10,
            17,18,19,20,21,22
        ]

        X["peak_hour"] = (
            X["hour_of_day"]
            .isin(peak_hours)
            .astype(int)
        )

        X["night_user"] = (
            (
                X["hour_of_day"] >= 22
            )
            |
            (
                X["hour_of_day"] <= 5
            )
        ).astype(int)

        X["weekend_evening"] = (
            (
                X["is_weekend"] == 1
            )
            &
            (
                X["hour_of_day"] >= 18
            )
        ).astype(int)

        # =====================================
        # Interaction Features
        # =====================================

        X["user_ad_affinity"] = (
            X["user_historical_ctr"]
            *
            X["ad_quality_score"]
        )

        X["interest_quality"] = (
            X["interest_match"]
            *
            X["ad_quality_score"]
        )

        X["user_page_affinity"] = (
            X["user_historical_ctr"]
            *
            X["page_quality_score"]
        )

        # =====================================
        # Behavioral Features
        # =====================================

        X["recent_clicker"] = (
            X["time_since_last_click"]
            < 3600
        ).astype(int)

        X["recent_impression"] = (
            X["time_since_last_impression"]
            < 1800
        ).astype(int)

        X["ad_fatigue"] = (
            X["sequence_position"]
            > 5
        ).astype(int)

        X["click_momentum"] = (
            X["previous_ad_clicks_today"]
            /
            (X["sequence_position"] + 1)
        )

        # =====================================
        # Seasonality Features
        # =====================================

        X["quarter"] = (
            (X["month"] - 1) // 3 + 1
        )

        X["month_sin"] = np.sin(
            2 * np.pi * X["month"] / 12
        )

        X["month_cos"] = np.cos(
            2 * np.pi * X["month"] / 12
        )

        return X