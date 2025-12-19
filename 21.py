import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(
    r"C:\Users\User\Desktop\學習2\student_lifestyle_dataset.csv"
)
df = df.drop(columns=["Student_ID"])


stress_map = {
    "Low": 1,
    "Moderate": 2,
    "High": 3
}
df["Stress_Level"] = df["Stress_Level"].map(stress_map)

X = df.drop(columns=["GPA"])
y = df["GPA"]

numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(exclude=["number"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== Random Forest GPA 預測結果 ===")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


result = pd.DataFrame({
    "Actual_GPA": y_test.values,
    "Predicted_GPA": y_pred
})
print(result.head(10))

preprocess_fitted = clf.named_steps["preprocess"]
rf_model = clf.named_steps["model"]

feature_names = preprocess_fitted.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("=== 前 10 個最重要特徵 ===")
print(importance_df.head(10))