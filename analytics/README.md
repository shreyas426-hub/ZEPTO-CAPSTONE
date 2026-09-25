\# Module 2: Analytics Pipeline



\## Setup and Execution

1\. Install dependencies: `pip install seaborn matplotlib pandas scikit-learn imbalanced-learn joblib`

2\. Run EDA pipeline: `python 01\_eda.py`

3\. Run Modeling pipeline: `python 02\_modeling.py`



\## Written Interpretations



\### Task 2: Missing Value Strategy

\* \*\*embarked \& embark\_town (0.22% missing):\*\* Below 5% threshold; dropped rows.

\* \*\*age (19.87% missing):\*\* Between 5% and 30% threshold; imputed using median value.

\* \*\*deck (77.10% missing):\*\* Exceeds 30% threshold; column dropped to avoid unreliable imputation.



\### Task 3: Skewness Conclusion

\* \*\*Fare Statistics:\*\* Mean = 51.84, Median = 14.45, Mode = 8.05.

\* \*\*Ordering:\*\* Mean > Median > Mode, confirming a strongly right-skewed distribution.



\### Task 4: Strongest Correlations

1\. \*\*pclass \& fare (-0.55):\*\* Strong negative correlation showing higher class passenger tickets cost significantly more.

2\. \*\*parch \& sibsp (0.41):\*\* Moderate positive correlation indicating passengers traveling with parents/children often traveled with siblings/spouses.



\### Task 5: Multivariate Data Story

1\. \*\*Sex \& Pclass:\*\* Women in 1st/2nd class had survival rates over 85%, whereas 3rd-class men had rates under 15%.

2\. \*\*Age \& Fare:\*\* Passengers paying higher fares had higher survival rates regardless of age.

3\. \*\*Family Size:\*\* Small families (2-4 members) had better survival odds than solo travelers or large families.

4\. \*\*Embarked Port:\*\* Cherbourg (C) departures showed higher survival due to a higher proportion of 1st-class passengers.



\### Task 11: Imbalance Strategy Conclusion

Class Weight Balancing and SMOTE slightly improved recall for the minority class (survived) but caused a drop in precision. Overall, the baseline model achieved the best balanced F1-score.



\### Task 13: Heteroscedasticity Conclusion

The residual plot displays clear heteroscedasticity (fanning out at higher predicted values), showing that variance of errors increases for higher ticket fares.



\## Model Comparison Table



| Classifier Model | Accuracy | Precision | Recall | F1 Score | AUC | Regression Metrics | Value |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| Logistic Regression | 0.8034 | 0.7727 | 0.7391 | 0.7556 | 0.8521 | MAE | 14.22 |

| Decision Tree | 0.7865 | 0.8036 | 0.6522 | 0.7200 | 0.8240 | RMSE | 26.85 |

| Random Forest (Tuned)| \*\*0.8202\*\*| \*\*0.8125\*\*| \*\*0.7536\*\*| \*\*0.7819\*\*| \*\*0.8680\*\*| R² / Adj R² | 0.521 / 0.498 |



\### Deployment Recommendation

We recommend deploying the \*\*Tuned Random Forest Classifier\*\*. It achieved the highest overall Accuracy (82.02%), F1 Score (0.7819), and ROC AUC (0.8680). Its ensemble structure effectively captures complex non-linear feature interactions without overfitting.

