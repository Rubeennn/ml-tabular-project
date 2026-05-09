import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from src.preprocessing.preprocessing import Preprocessing
from src.config.model_params import MODEL_PARAMS
import joblib


def load_data(path):
    return pd.read_csv(path)


def main():
    data = load_data("data/raw/insurance.csv"
    )

    X = data.drop("charges", axis=1)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = MODEL_PARAMS["gradient_boosting"]
    pipeline = Pipeline(
        [
            ("preprocessing", Preprocessing()),
            ("model", GradientBoostingRegressor(**params)),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    score = rmse(y_test, preds)

    print(f"RMSE: {score}")

    joblib.dump(pipeline, "artifacts/model.pkl")


if __name__ == "__main__":
    main()
