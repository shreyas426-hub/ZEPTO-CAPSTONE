import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# TASK 1: Single Load & Profiling
# ==========================================
print("--- TASK 1: DATA LOADING AND PROFILING ---")
# Load raw dataset once from seaborn
raw_df = sns.load_dataset('titanic')

# Save offline fallback immediately
raw_df.to_csv("titanic.csv", index=False)
print("Dataset loaded from network/cache and saved to titanic.csv fallback.")

print("\n--- INFO ---")
raw_df.info()

print("\n--- SHAPE ---")
print(f"Shape: {raw_df.shape}")

print("\n--- DESCRIBE ---")
print(raw_df.describe(include='all'))

print("\n--- MISSING VALUES PERCENTAGE ---")
missing = raw_df.isnull().mean() * 100
missing_cols = missing[missing > 0]
for col, pct in missing_cols.items():
    print(f"{col}: {pct:.2f}%")

# ==========================================
# TASK 2: Missing Value Handling Decisions
# ==========================================
print("\n--- TASK 2: MISSING VALUE STRATEGY ---")
df = raw_df.copy()
# 1. Drop 'deck' column (77.10% missing - too high for reliable imputation)
df.drop(columns=['deck'], inplace=True)

# 2. Impute 'age' with median (19.87% missing)
age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)

# 3. Drop rows with missing 'embarked' / 'embark_town' (0.22% missing)
df.dropna(subset=['embarked', 'embark_town'], inplace=True)

print(f"Cleaned DataFrame Shape: {df.shape}")

# ==========================================
# TASK 3: Univariate Analysis (Age & Fare)
# ==========================================
print("\n--- TASK 3: UNIVARIATE ANALYSIS ---")
for col in ['age', 'fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col} Outliers Count (IQR Rule): {len(outliers)}")

fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]

print(f"\nFare Metrics -> Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
print("Fare Distribution Conclusion: Right-skewed because Mean > Median > Mode.")

# Plot Age & Fare Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['age'], kde=True, ax=axes[0, 0]).set(title='Age Distribution')
sns.boxplot(x=df['age'], ax=axes[0, 1]).set(title='Age Boxplot')
sns.histplot(df['fare'], kde=True, ax=axes[1, 0]).set(title='Fare Distribution')
sns.boxplot(x=df['fare'], ax=axes[1, 1]).set(title='Fare Boxplot')
plt.tight_layout()
plt.savefig('univariate_age_fare.png')
plt.close()

# ==========================================
# TASK 4: Bivariate Analysis & Heatmap
# ==========================================
print("\n--- TASK 4: BIVARIATE ANALYSIS ---")
surv_sex = df.groupby('sex')['survived'].mean()
surv_pclass = df.groupby('pclass')['survived'].mean()
surv_sex_pclass = df.groupby(['sex', 'pclass'])['survived'].mean()

print("Survival Rate by Sex:\n", surv_sex)
print("\nSurvival Rate by Pclass:\n", surv_pclass)
print("\nSurvival Rate by Sex & Pclass:\n", surv_sex_pclass)

# Correlation Matrix (Restricted strictly to 6 numeric columns)
num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('6x6 Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Find top 2 off-diagonal absolute correlations
corr_abs_vals = corr_matrix.abs().to_numpy().copy()
np.fill_diagonal(corr_abs_vals, 0)
corr_abs = pd.DataFrame(corr_abs_vals, index=corr_matrix.index, columns=corr_matrix.columns)
top_pairs = corr_abs.unstack().sort_values(ascending=False).drop_duplicates().head(2)

print("\nTwo Strongest Feature Correlations (Off-Diagonal):")
for pair, val in top_pairs.items():
    actual_val = corr_matrix.loc[pair[0], pair[1]]
    print(f"- {pair[0]} & {pair[1]}: {actual_val:.2f}")

# ==========================================
# TASK 5: Multivariate Visual Data Story
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(data=df, x='pclass', y='survived', hue='sex', ax=axes[0, 0])
axes[0, 0].set_title('1. Survival Rate by Pclass and Sex')

sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title('2. Age vs Fare distribution colored by Survival')

df['family_size'] = df['sibsp'] + df['parch']
sns.barplot(data=df, x='family_size', y='survived', ax=axes[1, 0])
axes[1, 0].set_title('3. Survival Rate by Family Size')

sns.barplot(data=df, x='embarked', y='survived', hue='pclass', ax=axes[1, 1])
axes[1, 1].set_title('4. Survival Rate by Embarked Port and Class')

plt.tight_layout()
plt.savefig('multivariate_data_story.png')
plt.close()

# ==========================================
# TASK 6: Exploratory Standardization Check
# ==========================================
print("\n--- TASK 6: EDA STANDARDIZATION SANITY CHECK ---")
df_std = df.copy()
df_std['age_z'] = (df_std['age'] - df_std['age'].mean()) / df_std['age'].std()
df_std['fare_z'] = (df_std['fare'] - df_std['fare'].mean()) / df_std['fare'].std()

print(f"Age Before -> Mean: {df['age'].mean():.2f}, Std: {df['age'].std():.2f}")
print(f"Age Z-score -> Mean: {df_std['age_z'].mean():.2f}, Std: {df_std['age_z'].std():.2f}")
print(f"Fare Before -> Mean: {df['fare'].mean():.2f}, Std: {df['fare'].std():.2f}")
print(f"Fare Z-score -> Mean: {df_std['fare_z'].mean():.2f}, Std: {df_std['fare_z'].std():.2f}")

print("\n01_eda.py completed successfully!")