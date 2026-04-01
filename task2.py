import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv") 

print("\n===== BASIC INFO =====")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# -----------------------------
# HISTOGRAMS
# -----------------------------
df.hist(figsize=(12,10))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# -----------------------------
# BAR CHARTS (CATEGORICAL)
# -----------------------------
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# SCATTER PLOT
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns

if len(num_cols) >= 2:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
    plt.title(f"{num_cols[0]} vs {num_cols[1]}")
    plt.show()

# -----------------------------
# BOXPLOTS (OUTLIERS)
# -----------------------------
for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# -----------------------------
# FINAL INSIGHTS (AUTO SUMMARY)
# -----------------------------
print("\n===== KEY INSIGHTS =====")

# Skewness check
print("\nSkewness:\n", df.skew(numeric_only=True))

# Correlation insights
corr = df.corr(numeric_only=True)
high_corr = corr[(corr > 0.7) & (corr < 1.0)]
print("\nHigh Correlations (>0.7):\n", high_corr)

print("\nEDA COMPLETED SUCCESSFULLY 🚀")