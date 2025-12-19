import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(r"C:\Users\User\Desktop\學習2\student_lifestyle_dataset.csv")
df = df.drop(columns=["Student_ID"], errors="ignore")

stress_map = {"Low": 1, "Moderate": 2, "High": 3}
if "Stress_Level" in df.columns:
    df["Stress_Level"] = df["Stress_Level"].map(stress_map)

low_study, high_study = 8, 9

df_fixed = df[
    (df["Study_Hours_Per_Day"] >= low_study) &
    (df["Study_Hours_Per_Day"] <= high_study)
].copy()

print(f"固定讀書時間 {low_study}~{high_study} 小時後的樣本數：", len(df_fixed))


df_fixed["Low_GPA"] = (df_fixed["GPA"] < 3.0).astype(int)

X = df_fixed.drop(columns=["GPA", "Low_GPA"])
y = df_fixed["Low_GPA"]


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
    ],
    remainder="drop"
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() == 2 else None
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== 在固定讀書時間下預測 GPA 是否偏低 ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

rf_model = clf.named_steps["model"]
feature_names = clf.named_steps["preprocess"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df["Feature"] = importance_df["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)

print("\n=== 特徵重要度 Top 10（最適合放 PPT）===")
print(importance_df.head(10))
