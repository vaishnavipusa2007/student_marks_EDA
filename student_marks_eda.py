# Student Marks Dataset — Exploratory Data Analysis (EDA)
# Run this notebook with student_marks.csv in the same folder.
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_FILE = "student_marks.csv"
df = pd.read_csv(DATA_FILE)

# 1. Dataset overview
print("Original shape:", df.shape)
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values before cleaning:\n", df.isna().sum())
print("Duplicate rows before cleaning:", df.duplicated().sum())

# 2. Cleaning: remove duplicates, then fill numeric missing values with medians
df = df.drop_duplicates().copy()
for col in ["Study_Hours", "Exam_Marks"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# IQR outlier treatment for Study_Hours, matching the original notebook
Q1 = df["Study_Hours"].quantile(0.25)
Q3 = df["Study_Hours"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_count = ((df["Study_Hours"] < lower_bound) | (df["Study_Hours"] > upper_bound)).sum()
df = df[(df["Study_Hours"] >= lower_bound) & (df["Study_Hours"] <= upper_bound)].copy()
df["Study_Hours"] = df["Study_Hours"].clip(lower_bound, upper_bound)

print("\nRows after cleaning:", len(df))
print("Missing values after cleaning:\n", df.isna().sum())
print("Study_Hours IQR bounds:", round(lower_bound, 2), "to", round(upper_bound, 2))
print("Study_Hours outliers removed:", int(outlier_count))
print("\nDescriptive statistics:\n", df[["Study_Hours", "Exam_Marks"]].describe().round(2))
print("Correlation (Study_Hours vs Exam_Marks):", round(df["Study_Hours"].corr(df["Exam_Marks"]), 3))

# 3. Visual EDA — seven charts, each with a title and labelled axes
plt.figure(figsize=(7,4)); sns.histplot(df["Study_Hours"], bins=10, kde=True, color="#5279a8")
plt.title("Distribution of Study Hours"); plt.xlabel("Study Hours"); plt.ylabel("Number of Students"); plt.show()

plt.figure(figsize=(7,4)); sns.histplot(df["Exam_Marks"], bins=10, kde=True, color="#4d9b83")
plt.title("Distribution of Exam Marks"); plt.xlabel("Exam Marks"); plt.ylabel("Number of Students"); plt.show()

plt.figure(figsize=(7,4)); sns.regplot(data=df, x="Study_Hours", y="Exam_Marks", scatter_kws={"alpha":0.65})
plt.title("Study Hours vs Exam Marks"); plt.xlabel("Study Hours"); plt.ylabel("Exam Marks"); plt.show()

plt.figure(figsize=(7,3)); sns.boxplot(x=df["Study_Hours"])
plt.title("Study Hours: Box Plot"); plt.xlabel("Study Hours"); plt.show()

plt.figure(figsize=(7,3)); sns.boxplot(x=df["Exam_Marks"])
plt.title("Exam Marks: Box Plot"); plt.xlabel("Exam Marks"); plt.show()

plt.figure(figsize=(5,4)); sns.heatmap(df[["Study_Hours", "Exam_Marks"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap"); plt.show()

bins = [0, 2, 4, 6, 8, np.inf]
labels = ["0–2", "2–4", "4–6", "6–8", "8+"]
grouped = df.copy()
grouped["Study_Hour_Group"] = pd.cut(grouped["Study_Hours"], bins=bins, labels=labels, include_lowest=True)
mean_marks = grouped.groupby("Study_Hour_Group", observed=False)["Exam_Marks"].mean().dropna()
plt.figure(figsize=(7,4)); sns.barplot(x=mean_marks.index.astype(str), y=mean_marks.values, color="#7189bf")
plt.title("Average Exam Marks by Study-Hour Group"); plt.xlabel("Study-Hour Group"); plt.ylabel("Average Exam Marks"); plt.show()

# 4. Export the cleaned dataset and print group averages for evidence-based findings
df.to_csv("cleaned_student_marks.csv", index=False)
print("\nAverage marks by study-hour group:\n", mean_marks.round(2))
print("Cleaned dataset saved as cleaned_student_marks.csv")
print("Note: check whether marks outside the expected exam scale need investigation before final submission.")
