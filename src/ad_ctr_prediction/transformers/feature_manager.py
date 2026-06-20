class FeatureManager:

    TARGET_COLUMN = "click"

    IDENTIFIER_COLUMNS = [

        "user_id",

        "ad_id",

        "advertiser_id",

        "timestamp"
    ]

    HIGH_CARDINALITY_COLUMNS = [

        "user_browser",

        "user_os",

        "user_location_region",

        "ad_category",

        "page_category"
    ]

    LOW_CARDINALITY_COLUMNS = [

        "user_gender",

        "user_age_group",

        "user_device_type",

        "ad_position",

        "ad_creative_type",

        "ad_size",

        "seasonality"
    ]