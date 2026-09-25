import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, mean_absolute_error, 
                             mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Read committed fallback file
print("--- TASK 7: READ COMMITTED DATA & STRATIFIED SPLIT ---")
df = pd.read_csv("titanic.csv")

X = df.drop(columns=['survived', 'alive', 'who', 'adult_male', 'alone', 'deck'], errors='ignore')
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Preprocessing Transformers
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass', 'class', 'embark_town']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit preprocessor ONLY on train data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Classifiers
print("\n--- TASK 9 & 10: CLASSIFIER EVALUATION ---")
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []
plt.figure(figsize=(8, 6))

for name, clf in classifiers.items():
    clf.fit(X_train_prep, y_train)
    y_pred = clf.predict(X_test_prep)
    y_proba = clf.predict_proba(X_test_prep)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

print(pd.DataFrame(results).to_string(index=False))

# Decision Tree Plot
plt.figure(figsize=(15, 8))
plot_tree(classifiers['Decision Tree'], filled=True, feature_names=preprocessor.get_feature_names_out(), class_names=['Not Survived', 'Survived'])
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig('decision_tree.png')
plt.close()

# Imbalance comparison
print("\n--- TASK 11: IMBALANCE HANDLING ---")
lr_base = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_prep, y_train)
y_pred_base = lr_base.predict(X_test_prep)

lr_bal = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X_train_prep, y_train)
y_pred_bal = lr_bal.predict(X_test_prep)

smote = SMOTE(random_state=42)
X_tr_smote, y_tr_smote = smote.fit_resample(X_train_prep, y_train)
lr_smote = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_smote, y_tr_smote)
y_pred_smote = lr_smote.predict(X_test_prep)

imb_data = [
    {'Strategy': 'Baseline', 'Precision': precision_score(y_test, y_pred_base), 'Recall': recall_score(y_test, y_pred_base), 'F1': f1_score(y_test, y_pred_base)},
    {'Strategy': 'Class Weight Balanced', 'Precision': precision_score(y_test, y_pred_bal), 'Recall': recall_score(y_test, y_pred_bal), 'F1': f1_score(y_test, y_pred_bal)},
    {'Strategy': 'SMOTE Oversampling', 'Precision': precision_score(y_test, y_pred_smote), 'Recall': recall_score(y_test, y_pred_smote), 'F1': f1_score(y_test, y_pred_smote)}
]
print(pd.DataFrame(imb_data).to_string(index=False))

# Tuning
print("\n--- TASK 12: GRIDSEARCHCV & OOB ---")
rf_oob = RandomForestClassifier(oob_score=True, random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'max_features': ['sqrt', 'log2']}
grid = GridSearchCV(rf_oob, param_grid, cv=5, scoring='f1').fit(X_train_prep, y_train)
best_rf = grid.best_estimator_

print(f"Best Hyperparameters: {grid.best_params_}")
print(f"OOB Score: {best_rf.oob_score_:.4f}")

# Regression task
print("\n--- TASK 13: REGRESSION SIDE-TASK ---")
X_reg = df.drop(columns=['fare', 'deck'], errors='ignore')
y_reg = df['fare']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_num_cols = ['age', 'sibsp', 'parch']
reg_cat_cols = ['sex', 'embarked', 'pclass', 'class', 'embark_town']

reg_prep = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), reg_num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), reg_cat_cols)
])

X_tr_r_p = reg_prep.fit_transform(X_tr_r)
X_te_r_p = reg_prep.transform(X_te_r)

reg_model = LinearRegression().fit(X_tr_r_p, y_tr_r)
y_pred_reg = reg_model.predict(X_te_r_p)

mae = mean_absolute_error(y_te_r, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_reg))
r2 = r2_score(y_te_r, y_pred_reg)
n, k = X_te_r_p.shape[0], X_te_r_p.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}")

# Residual plot
residuals = y_te_r - y_pred_reg
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_reg, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residuals_plot.png')
plt.close()

# Save complete pipeline artifact
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_rf)
])
full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, 'full_analytics_pipeline.joblib')

# Reload and test
reloaded_pipeline = joblib.load('full_analytics_pipeline.joblib')
sample_preds = reloaded_pipeline.predict(X_test.iloc[:2])
print(f"Reload verification sample predictions: {sample_preds}")

print("\n02_modeling.py completed successfully!")