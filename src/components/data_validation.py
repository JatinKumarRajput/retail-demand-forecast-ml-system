import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    processed_data_path: str = "data/processed/processed_sales.csv"

class DataValidation:

    def __init__(self,config:DataValidationConfig):
        self.config=config

    def vaildate_data(self):
        print("Starting Data Validation...")

        if not os.path.exists(self.config.processed_data_path):
            raise Exception("Processed dataset not found!")
        
        df=pd.read_csv(self.config.processed_data_path)

        print("Dataset loaded for validation.")

        # Check missing values
        missing_values=df.isnull().sum().sum()

        if missing_values>0:
            print("Warning: Dataset contains missing values!")
        else:
            print("No missing values found.")

        # Check for duplicates
        duplicates=df.duplicated().sum()

        if duplicates>0:
            print(f"Warning: Dataset contains {duplicates} duplicate rows.")
        else:
            print("No duplicate rows found.")
        
        print("Data Validation Completed.")

        return True
