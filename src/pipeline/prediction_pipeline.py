import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):
        self.model_path = "models/best_model.pkl"

        # Load trained model
        self.model = joblib.load(self.model_path)

        # Expected feature columns (same as training)
        self.expected_columns = [
            'id','store_nbr','onpromotion','day_of_week','month','year','is_weekend',
            'lag_1','lag_7','rolling_mean_7',
            'family_AUTOMOTIVE','family_BABY CARE','family_BEAUTY','family_BEVERAGES',
            'family_BOOKS','family_BREAD/BAKERY','family_CELEBRATION','family_CLEANING',
            'family_DAIRY','family_DELI','family_EGGS','family_FROZEN FOODS',
            'family_GROCERY I','family_GROCERY II','family_HARDWARE',
            'family_HOME AND KITCHEN I','family_HOME AND KITCHEN II',
            'family_HOME APPLIANCES','family_HOME CARE','family_LADIESWEAR',
            'family_LAWN AND GARDEN','family_LINGERIE','family_LIQUOR,WINE,BEER',
            'family_MAGAZINES','family_MEATS','family_PERSONAL CARE',
            'family_PET SUPPLIES','family_PLAYERS AND ELECTRONICS','family_POULTRY',
            'family_PREPARED FOODS','family_PRODUCE',
            'family_SCHOOL AND OFFICE SUPPLIES','family_SEAFOOD'
        ]

    def predict(self, input_data: dict):

        df = pd.DataFrame([input_data])

        # Add missing columns
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Arrange column order
        df = df[self.expected_columns]

        prediction = self.model.predict(df)

        return prediction[0]
