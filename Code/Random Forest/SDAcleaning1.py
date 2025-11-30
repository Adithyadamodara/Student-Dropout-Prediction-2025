## MAIN GOAL: CLEAN DATASET AND PREPARE IT FOR MODELING
# TASKS :
# 1. Merge datasets (studentInfo, studentRegistration, studentVle)
# 2. Create target variable (Withdrawn vs Not Withdrawn)
# 3. Feature Engineering (Total Clicks from VLE dataset)
# 4. Handle Missing Values
# 5. Drop columns that may cause Data Leakage
# 6. Final check of cleaned dataset


import pandas as pd 

info = pd.read_csv('studentInfo.csv')
reg = pd.read_csv('studentRegistration.csv')
vle = pd.read_csv('studentVle.csv')

df = info.merge(reg, on=['id_student', 'code_module','code_presentation'])

# Creating target variable (Dataset has 4 varaiable, PASS, FAIL, DISTINCTION, WITHDRAWN)
# 1: Withdrawn, 0: Not Withdrawn (PASS, FAIL, DISTINCTION)

df['target'] = df['final_result'].apply(lambda x: 1 if x=='Withdrawn' else 0)
print(df['target'].value_counts())

# First feature from VLE dataset: Total Number of clisks per student, this feature
# indicates the engagement of student with online learning platform
# BUT column has multiple entries per student, so we need to aggregate it
# We will calculate total clicks per student and merge it with main dataframe
vle_agg = vle.groupby('id_student')['sum_click'].sum().reset_index()
vle_agg.columns = ['id_student', 'total_clicks']

# Merging aggregated VLE data with main dataframe
df = df.merge(vle_agg, on='id_student', how='left')
df['total_clicks'] = df['total_clicks'].fillna(0) # Replacing NaN values with 0 clicks

# Drop columns that cause Data Leakage
df = df.drop(columns=['final_result', 'id_student', 'date_unregistration'], axis=1)
# Data registration date could be useful, so we keep it for now

# Handling missing values for numerical columns
df['imd_band'].fillna('Unknown',inplace=True) # Categorical column 
df['date_registration'].fillna(df['date_registration'].median(),inplace=True) # Numerical column

df.info()

# Saving Dataset
df.to_csv('cleaned_student_data.csv', index=False)
