import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn

# Fix MLflow path issue on Windows
mlflow.set_tracking_uri("file:./mlruns")


@dataclass
class ModelTrainerConfig:
    feature_data_path: str = "data/processed/feature_engineered_data.csv"
    model_path: str = "models/best_model.pkl"


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):

        print("Starting Model Training...")

        df = pd.read_csv(self.config.feature_data_path)

        # Sample dataset for faster training
        df = df.sample(300000, random_state=42)

        # Feature-target split
        X = df.drop(columns=["sales", "date"])
        y = df["sales"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define models
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=20,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),

            "GradientBoosting": GradientBoostingRegressor(),

            "XGBoost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1
            )
        }

        model_scores = {}
        trained_models = {}

        # Train models
        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                print(f"Training {name}...")

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, predictions))

                print(f"{name} RMSE: {rmse}")

                # Log parameters
                mlflow.log_param("model_name", name)

                # Log metric
                mlflow.log_metric("rmse", rmse)

                # Log model artifact
                mlflow.sklearn.log_model(model, name)

                model_scores[name] = rmse
                trained_models[name] = model

        # Select best model
        best_model_name = min(model_scores, key=model_scores.get)
        best_rmse = model_scores[best_model_name]
        best_model = trained_models[best_model_name]

        print(f"\nBest Model: {best_model_name}")
        print(f"Best RMSE: {best_rmse}")

        # Hyperparameter tuning for XGBoost

        if best_model_name == "XGBoost":

            print("\nStarting Hyperparameter Tuning for XGBoost...")
            param_grid = {
                "n_estimators": [100,200,300],
                "max_depth": [4,6,8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
            
            xgb = XGBRegressor(n_jobs=-1)

            random_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_grid,
                n_iter=10,
                scoring="neg_root_mean_squared_error",
                cv=3,
                verbose=1,
                n_jobs=-1
            )

            random_search.fit(X_train, y_train)

            tuned_model = random_search.best_estimator_

            predictions = tuned_model.predict(X_test)

            tuned_rmse = np.sqrt(mean_squared_error(y_test, predictions))

            print(f"Tuned XGBoost RMSE: {tuned_rmse}")
            print(f"Best Hyperparameters: {random_search.best_params_}")

            # Log tuning results in MLflow
            with mlflow.start_run(run_name="XGBoost_HyperTuning"):

                mlflow.log_params(random_search.best_params_)
                mlflow.log_metric("tuned_rmse", tuned_rmse)
                mlflow.sklearn.log_model(tuned_model, "XGBoost_Tuned")
            
            best_model = tuned_model
            best_rmse = tuned_rmse

        os.makedirs("models", exist_ok=True)

        joblib.dump(best_model, self.config.model_path)

        print("Best model saved successfully.")

        return self.config.model_path