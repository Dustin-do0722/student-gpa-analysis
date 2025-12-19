import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"C:\Users\User\Desktop\學習2\student_lifestyle_dataset.csv")


df = df.drop(columns=["Student_ID"])

stress_map = {"Low": 1, "Moderate": 2, "High": 3}
df["Stress_Level"] = df["Stress_Level"].map(stress_map)

features = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Stress_Level"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_profile = df.groupby("Cluster")[features + ["GPA"]].mean()

print("=== 各群學生特徵與 GPA 平均 ===")
print(cluster_profile)

print("\n=== 各群學生人數 ===")
print(df["Cluster"].value_counts())

print("\n=== 群內壓力對 GPA 的影響（平均值） ===")

stress_gpa_summary = (
    df.groupby(["Cluster", "Stress_Level"])
      .agg(
          mean_GPA=("GPA", "mean"),
          std_GPA=("GPA", "std"),
          count=("GPA", "count")
      )
      .reset_index()
      .sort_values(["Cluster", "Stress_Level"])
)

print(stress_gpa_summary)

print("\n=== 群內壓力對 GPA 的線性關係（斜率與 R²） ===")

for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    x = sub["Stress_Level"].to_numpy(dtype=float)
    y = sub["GPA"].to_numpy(dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    intercept = y_mean - slope * x_mean

    y_hat = intercept + slope * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    print(
        f"Cluster {c}: "
        f"slope(Stress→GPA) = {slope:.4f}, "
        f"R² = {r2:.4f}, "
        f"n = {len(sub)}"
    )
