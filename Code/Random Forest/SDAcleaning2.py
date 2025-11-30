## GOAL: To engineer new features to get weekly engagement insights for each student
# Since VLE engagement is the most important predictor its necessary to correctly utilize
# weekly data to get the best insights for the model

import pandas as pd 

info = pd.read_csv('studentInfo.csv')
reg = pd.read_csv('studentRegistration.csv')
vle = pd.read_csv('studentVle.csv')

# Initial merge
df = info.merge(reg, on=['id_student', 'code_module','code_presentation'])

# Cleaning
df['target'] = df['final_result'].apply(lambda x: 1 if x=='Withdrawn' else 0)

# Convereting day to week numbers
vle['week'] = (vle['date'] // 7).astype(int)

# Using segmented weeks to calculate engagment of student 
# reset_index to reframe dataframe by index after aggrgation
weekly_clicks = vle.groupby(['id_student','week'])['sum_click'].sum().reset_index()

# Pivoting weekly data, each week becomes a column
weekly_pivot = weekly_clicks.pivot(index='id_student', columns='week', values='sum_click').fillna(0)

# Renaming columns for clarity
weekly_pivot.columns = [f'week_{col}_clicks' for col in weekly_pivot.columns]

# Summarizing engagement patterns
weekly_pivot['num_active_weeks'] = (weekly_pivot > 0).sum(axis=1)
weekly_pivot['click_trend'] = weekly_pivot.diff(axis=1).mean(axis=1) # slope of engagement

# Merging enginnered weekly_pivot columns with main dataset
df = df.merge(weekly_pivot, left_on='id_student', right_index=True, how='left')
df.fillna(0, inplace=True)

df = df.drop(columns=['id_student','final_result','date_unregistration'], axis=1)

df['imd_band'].fillna('Unknown',inplace=True)
df['date_registration'].fillna(df['date_registration'].median(),inplace=True)

df.info()

df.to_csv('cleaned_dataset_2.csv', index=False)
print("New cleaned csv file created successfully!")