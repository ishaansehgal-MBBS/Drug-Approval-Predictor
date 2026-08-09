import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# 1. Load Data
df = pd.read_csv('ClinicalTrialData.csv')

# Step 2: Calculate baseline success rate
success_pct = (df['outcome'].sum() / len(df)) * 100
print(f"Percentage of successful trials: {success_pct:.2f}%")

# Step 3: Feature selection (drop 'tid')
df = df.drop(columns=['tid'])

# Step 4: MinMax Normalization on 'taracc'
scaler = MinMaxScaler()
df['taracc'] = scaler.fit_transform(df[['taracc']])

# Separate features (X) and target (y)
X = df.drop(columns=['outcome'])
y = df['outcome']

# Step 5: Stratified Split (70% train, 30% test, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 6 & 7: Baseline Models (Logistic Regression vs Decision Forest)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"Baseline Logistic Regression AUC: {lr_auc:.4f}")
print(f"Baseline Decision Forest AUC: {rf_auc:.4f}")

# Step 9: Feature Importance for Logistic Regression
weights = pd.Series(lr.coef_[0], index=X.columns)
top_5_features = weights.abs().sort_values(ascending=False).head(5)
print("\nTop 5 Most Important Features (Logistic Regression):")
print(top_5_features)

top5 = weights.abs().sort_values(ascending=False).head(5).index
print(pd.DataFrame({
    "feature": top5,
    "coefficient": weights[top5],
    "absolute_weight": weights[top5].abs()
}).sort_values("absolute_weight", ascending=False))

# Step 10 & 11: Hyperparameter Tuning with 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tune Logistic Regression
lr_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr_tuned = GridSearchCV(LogisticRegression(max_iter=1000), lr_grid, scoring='roc_auc', cv=cv)
lr_tuned.fit(X_train, y_train)
tuned_lr_auc = roc_auc_score(y_test, lr_tuned.predict_proba(X_test)[:, 1])

# Tune Decision Forest (Random Forest)
rf_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
rf_tuned = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, scoring='roc_auc', cv=cv)
rf_tuned.fit(X_train, y_train)
tuned_rf_auc = roc_auc_score(y_test, rf_tuned.predict_proba(X_test)[:, 1])

print(f"\nTuned Logistic Regression AUC: {tuned_lr_auc:.4f}")
print(f"Tuned Decision Forest AUC: {tuned_rf_auc:.4f}")
