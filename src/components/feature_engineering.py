import pandas as pd
from dataclasses import dataclass

@dataclass
class FeatureEngineeringConfig:
    processed_data_path: str = "data/processed/processed_sales.csv"
    feature_data_path: str = "data/processed/feature_engineered_data.csv"

class FeatureEngineering:

    def __init__(self, config:FeatureEngineeringConfig):
        self.config=config

    def initiate_feature_engineering(self):

        print("Starting Feature Engineering...")

        df=pd.read_csv(self.config.processed_data_path)

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        #Create time-based features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

        # Lag features
        df = df.sort_values(by=["store_nbr", "family", "date"])

        df["lag_1"] = df.groupby(["store_nbr", "family"])["sales"].shift(1)
        df["lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)

        # Rolling mean
        df["rolling_mean_7"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        # One Hot Encoding for product family
        df = pd.get_dummies(df, columns=["family"])
        
        df = df.dropna()

        #Save feature engineered dataset
        df.to_csv(self.config.feature_data_path,index=False)

        print("Feature Engineering completed.")
        print("Feature dataset saved.")

        return self.config.feature_data_path