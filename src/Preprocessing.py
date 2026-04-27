from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class Preprocessing:
    def __init__(self):
        self.X_train = None
        self.y_train = None

        self.scaler = None
        self.encoder = None

    def fit(self, X_train, y_train=None):

        X_train = X_train.copy()

        X_train["bmi_category"] = X_train["bmi"].apply(self._bmi_category)
        X_train = self._add_age_group(X_train)
        X_train = self._add_smoker_age(X_train)
        X_train = self._add_smoker_bmi(X_train)

        self.cat_cols = ["sex", "region", "smoker", "age_group", "bmi_category"]
        self.num_cols = ["age", "bmi", "smoker_age", "smoker_bmi"]


        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(X_train[self.cat_cols])
        self.scaler = StandardScaler()
        self.scaler.fit(X_train[self.num_cols])

        return self

    def transform(self, X_test):

        X_test = X_test.copy()
        X_test = X_test.reset_index(drop=True)

        X_test["bmi_category"] = X_test["bmi"].apply(self._bmi_category)
        X_test = self._add_age_group(X_test)
        X_test = self._add_smoker_age(X_test)
        X_test = self._add_smoker_bmi(X_test)

        cat_cols = ["sex", "region", "smoker", "age_group", "bmi_category"]
        X_cat = X_test[cat_cols]

        X_cat_encoded = pd.DataFrame(
            self.encoder.transform(X_cat),
            columns=self.encoder.get_feature_names_out(cat_cols),
            index=X_test.index
        )

        num_cols = ["age", "bmi", "smoker_age", "smoker_bmi"]
        X_num = X_test[num_cols]


        X_test = pd.concat([X_num, X_cat_encoded], axis=1)


        return X_test

    def fit_transform(self, X_train, y_train=None):
        return self.fit(X_train, y_train).transform(X_train)


    @staticmethod
    def _bmi_category(bmi):
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"

    @staticmethod
    def _add_age_group(X):
        X["age_group"] = pd.cut(
            x=X["age"],
            bins=[18, 30, 45, 60],
            labels=["18_30", "30_45", "45_60", "60_100"],
            include_lowest=True,
        )
        return X

    @staticmethod
    def _add_smoker_age(X):
        X["smoker_age"] = (X["smoker"] == "yes").astype(int) * X["age"]
        return X

    @staticmethod
    def _add_smoker_bmi(X):
        X["smoker_bmi"] = (X["smoker"] == "yes").astype(int) * X["bmi"]
        return X
