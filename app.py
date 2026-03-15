from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

pipeline = PredictionPipeline()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    family = request.form["family"]

    data = {
        "id": int(request.form["id"]),
        "store_nbr": int(request.form["store_nbr"]),
        "onpromotion": int(request.form["onpromotion"]),
        "day_of_week": int(request.form["day_of_week"]),
        "month": int(request.form["month"]),
        "year": int(request.form["year"]),
        "is_weekend": int(request.form["is_weekend"]),
        "lag_1": float(request.form["lag_1"]),
        "lag_7": float(request.form["lag_7"]),
        "rolling_mean_7": float(request.form["rolling_mean_7"]),
    }

    # initialize all family columns as 0
    families = [
        "AUTOMOTIVE","BABY CARE","BEAUTY","BEVERAGES","BOOKS","BREAD/BAKERY",
        "CELEBRATION","CLEANING","DAIRY","DELI","EGGS","FROZEN FOODS",
        "GROCERY I","GROCERY II","HARDWARE","HOME AND KITCHEN I",
        "HOME AND KITCHEN II","HOME APPLIANCES","HOME CARE","LADIESWEAR",
        "LAWN AND GARDEN","LINGERIE","LIQUOR,WINE,BEER","MAGAZINES",
        "MEATS","PERSONAL CARE","PET SUPPLIES","PLAYERS AND ELECTRONICS",
        "POULTRY","PREPARED FOODS","PRODUCE","SCHOOL AND OFFICE SUPPLIES",
        "SEAFOOD"
    ]

    for f in families:
        data[f"family_{f}"] = 0

    data[f"family_{family}"] = 1

    prediction = pipeline.predict(data)

    # Business insight
    if prediction > data["rolling_mean_7"]:
        insight = "Demand likely to increase — consider increasing inventory."
    else:
        insight = "Demand stable or lower — current stock levels should be sufficient."

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sales: {prediction:.2f}",
        insight_text=insight
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)