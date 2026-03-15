from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.components.data_validation import DataValidation,DataValidationConfig
from src.components.feature_engineering import FeatureEngineering,FeatureEngineeringConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

if __name__ == "__main__":

    # Data Ingestion
    ingestion_config = DataIngestionConfig()

    ingestion = DataIngestion(ingestion_config)

    processed_path=ingestion.initiate_data_ingestion()

    print("Data Ingestion Completed")
    print("Processed file saved at:",processed_path)

    # Data Validation
    validation_config=DataValidationConfig()
    validator= DataValidation(validation_config)

    validator.vaildate_data()

    # Feature Engineering
    feature_config=FeatureEngineeringConfig()
    feature_engineer=FeatureEngineering(feature_config)

    feature_engineer.initiate_feature_engineering()

    # Model Training
    trainer_config=ModelTrainerConfig()
    trainer=ModelTrainer(trainer_config)
    trainer.initiate_model_training()
