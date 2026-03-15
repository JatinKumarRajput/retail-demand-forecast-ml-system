import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw/train.csv"
    processed_data_path: str = "data/processed/processed_sales.csv"

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        print("Starting Data Ingestion...")

        if not os.path.exists(self.config.raw_data_path):
            raise Exception("Dataset not found in data/raw")
        
        df=pd.read_csv(self.config.raw_data_path)

        print("Dataset loaded successfully.")

        os.makedirs(os.path.dirname(self.config.processed_data_path),exist_ok=True)

        df.to_csv(self.config.processed_data_path,index=False)

        print("Data saved to processed folder")

        return self.config.processed_data_path

