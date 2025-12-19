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
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

cluster_profile = df.groupby("Cluster")[features + ["GPA"]].mean()
print("=== 各群學生特徵與 GPA 平均 ===")
print(cluster_profile)


print("\n=== 各群學生人數 ===")
print(df["Cluster"].value_counts())