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



print("=== 各群學生特徵與 GPA 平均 ===")
print(df.groupby("Cluster")[features + ["GPA"]].mean())

print("\n=== 各群學生人數 ===")
print(df["Cluster"].value_counts())


df["Social_Stress"] = df["Social_Hours_Per_Day"] * df["Stress_Level"]

factors = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Stress_Level",        
    "Social_Stress"        
]

print("\n=== 群內：生活型態因素（含壓力與交互作用）對 GPA 的影響 ===")

results = []

for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    y = sub["GPA"].to_numpy(dtype=float)
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()

    for f in factors:
        x = sub[f].to_numpy(dtype=float)
        x_mean = x.mean()

        if ((x - x_mean) ** 2).sum() == 0:
            continue

        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        intercept = y_mean - slope * x_mean

        y_hat = intercept + slope * x
        ss_res = ((y - y_hat) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        results.append({
            "Cluster": c,
            "Factor": f,
            "Slope": slope,
            "R2": r2,
            "Sample_Size": len(sub)
        })

result_df = pd.DataFrame(results)

for c in sorted(result_df["Cluster"].unique()):
    print(f"\n--- Cluster {c}（依 R² 排序）---")
    print(
        result_df[result_df["Cluster"] == c]
        .sort_values("R2", ascending=False)
    )
