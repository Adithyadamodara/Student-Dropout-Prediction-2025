import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Imbpipe
import numpy as np
import matplotlib.pyplot as plt
import shap

# Load Cleaned Dataset

data = pd.read_csv('cleaned_student_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# One-Hot Encoding for Categorical Variables
ordinal_cols = ['highest_education', 'age_band']
ordinal_categories = [
    ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
     'HE Qualification', 'Post Graduate Qualification'],
    ['0-35', '35-55', '55<=']
]

nominal_cols = ['code_module', 'code_presentation', 'gender', 'region', 'disability', 'imd_band']

num_cols = ['num_of_prev_attempts', 'studied_credits', 'date_registration', 'total_clicks']

preprocess = ColumnTransformer(
    transformers = [
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_cols),
    ],
    remainder='passthrough' # keep numerical columns as it is
)


# Hyperparameter Tuning and Model Training
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42
)

# New Imbalanced Pipeline with SMOTE
model = Imbpipe([
    ('preprocess', preprocess),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Classification Report:\n", classification_report(y_test, y_pred))


# Initial test only gives 80% accuracy and 100% on training data, indicating overfitting.
# Further hyperparameter tuning and cross-validation needed to improve model performance.

# Second test with SMOTE to handle class imbalance only gave 84% training and 80% testing accuracy.
# Shows importance of feature engineering and selection.

# Visualizaiton 

ohe = model.named_steps['preprocess'].named_transformers_['nom']
ord_enc = model.named_steps['preprocess'].named_transformers_['ord']

nominal_feature_names = ohe.get_feature_names_out(nominal_cols)
ordinal_feature_names = ordinal_cols  # same length after encoding
numeric_feature_names = num_cols

all_features = np.concatenate([
    ordinal_feature_names,
    nominal_feature_names,
    numeric_feature_names
])

# Get feature importance from the classifier
importances = model.named_steps['classifier'].feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(12,6))
plt.bar(range(top_n), importances[indices][:top_n])
plt.xticks(range(top_n), all_features[indices][:top_n], rotation=75, ha='right')
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# Create SHAP explainer for tree model
explainer = shap.TreeExplainer(model.named_steps['classifier'])

# Compute SHAP values
X_train_transformed = model.named_steps['preprocess'].transform(X_train)
shap_values = explainer.shap_values(X_train_transformed)

# Summary plot for class 1 (dropout)
shap.summary_plot(shap_values[1], X_train_transformed, 
                feature_names=all_features,
                show=False)


# Most Impactful Features from SHAP:
#1. total_clicks (Huge margin) 0.35
#2. highes education () 0.05

# This confirms that engagement (total clicks is the most important factor in predicting student dropout.)

# Need to re clean the data to retrieve more information on clicks and engagement.
