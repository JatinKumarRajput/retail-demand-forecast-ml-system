from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

sample_input = {
    "id":1,
    "store_nbr":1,
    "onpromotion":0,
    "day_of_week":2,
    "month":6,
    "year":2017,
    "is_weekend":0,
    "lag_1":220,
    "lag_7":210,
    "rolling_mean_7":215,
    "family_BEVERAGES":1
}

prediction = pipeline.predict(sample_input)

print("Predicted Sales:", prediction)