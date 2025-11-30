import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbpipe
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('cleaned_dataset_2.csv')

# Variables
X = data.drop('target', axis=1)
y = data['target']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# One-Hot Encoding for Categorical Variables
ordinal_cols = ['highest_education', 'age_band']
ordinal_categories = [
    ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
     'HE Qualification', 'Post Graduate Qualification'],
    ['0-35', '35-55', '55<=']
]

nominal_cols = ['code_module', 'code_presentation', 'gender', 'region', 'disability', 'imd_band']

num_cols = ['num_of_prev_attempts', 'studied_credits', 'date_registration', 'total_clicks'] + \
            [col for col in data.columns if col.startswith('week_')] + \
            ['num_active_weeks', 'click_trend']

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
    class_weight='balanced_subsample',
    random_state=42
)

# Imbalanced Pipeline with SMOTE
model = imbpipe([
    ('preprocess', preprocess),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Dropout', 'Dropout'])

plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Dropout Prediction")
plt.show()


# ROC AUC 
fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr,tpr)
# ROC curve 
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], 'r--') # Diagonal reference line
plt.xlabel("False Posiive Rate (FPR)")
plt.ylabel("True Posiive Rate (TPR)")
plt.title("ROC curve")
plt.grid(True)
plt.show()
# Getting Top 25 important features after model training

# Retrieving statistics
#ohe = model.named_steps['preprocess'].named_transformers_['nom']
#ord_enc = model.named_steps['preprocess'].named_transformers_['ord']

# Create feature name list
#nominal_feature_names = ohe.get_feature_names_out(nominal_cols)
#ordinal_feature_names = ordinal_cols
#numeric_feature_names = num_cols

#all_features = np.concatenate([
#    ordinal_feature_names,
#    nominal_feature_names,
#    numeric_feature_names
#])

#importances = model.named_steps['classifier'].feature_importances_
#indices = np.argsort(importances)[::-1]  # sort descending

#top_n = 25
#plt.figure(figsize=(14, 7))
#plt.bar(range(top_n), importances[indices][:top_n])
#plt.xticks(range(top_n),
#           all_features[indices][:top_n],
#           rotation=75,
#          ha='right')
#plt.title("Top 25 Feature Importances after Weekly Feature Engineering")
#plt.tight_layout()
#plt.show()


