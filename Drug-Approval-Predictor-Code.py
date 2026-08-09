import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Load Datasets
df = pd.read_csv('ClinicalTrialData.csv')
features_meta = pd.read_csv('Features.csv')

# 2. Calculate baseline success rate
success_pct = (df['outcome'].sum() / len(df)) * 100
print(f"Percentage of successful trials: {success_pct:.2f}%")

# 3. Feature & Target Separation
X = df.drop(columns=['tid', 'outcome'])
y = df['outcome']

# 4. Stratified Split (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Feature Scaling (Fit only on training data, preventing data leakage)
scaler = MinMaxScaler()
X_train['taracc'] = scaler.fit_transform(X_train[['taracc']])
X_test['taracc'] = scaler.transform(X_test[['taracc']])

# 6. Baseline Models (Logistic Regression vs Random Forest)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"Baseline Logistic Regression AUC: {lr_auc:.4f}")
print(f"Baseline Decision Forest AUC:     {rf_auc:.4f}")

# 7. Extract & describe feature importances (Logistic Regression Weights)
weights = pd.Series(lr.coef_[0], index=X.columns)
top_5_idx = weights.abs().sort_values(ascending=False).head(5).index

top_5_df = pd.DataFrame({
    "Categories": top_5_idx,
    "coefficient": weights[top_5_idx],
    "abs_weight": weights[top_5_idx].abs()
})

# Merge with metadata to get clean feature names and descriptions
readable_top_5 = top_5_df.merge(
    features_meta[['Categories', 'Feature', 'Description.1']], 
    on='Categories', 
    how='left'
)

# Combine Feature category and sub-description for clear human display
readable_top_5['Description'] = (
    readable_top_5['Feature'] + " (" + readable_top_5['Description.1'].fillna('') + ")"
)

display_df = readable_top_5[['Categories', 'Description', 'coefficient', 'abs_weight']].sort_values("abs_weight", ascending=False)

print("\nTop 5 Most Important Features (Logistic Regression):")
print(display_df.to_string(index=False))

# 8. Hyperparameter Tuning with 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tune Logistic Regression
lr_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr_tuned = GridSearchCV(LogisticRegression(max_iter=1000), lr_grid, scoring='roc_auc', cv=cv)
lr_tuned.fit(X_train, y_train)
tuned_lr_auc = roc_auc_score(y_test, lr_tuned.predict_proba(X_test)[:, 1])

# Tune Random Forest
rf_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
rf_tuned = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, scoring='roc_auc', cv=cv)
rf_tuned.fit(X_train, y_train)
tuned_rf_auc = roc_auc_score(y_test, rf_tuned.predict_proba(X_test)[:, 1])

print(f"\nTuned Logistic Regression AUC: {tuned_lr_auc:.4f}")
print(f"Tuned Decision Forest AUC:     {tuned_rf_auc:.4f}")
