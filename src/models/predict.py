import pandas as pd
import joblib


def main():
    model = joblib.load("artifacts/model.pkl")

    data = pd.read_csv("data/raw/insurance.csv")

    predictions = model.predict(data)

    data["prediction"] = predictions
    data.to_csv("artifacts/predictions.csv", index=False)


if __name__ == "__main__":
    main()
