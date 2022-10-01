# Load Data
import pandas as pd
df = pd.read_csv('/Users/emmaho/Downloads/archive/allegations_202007271729.csv')
df.head()
import numpy as np

# Data Preparation - Data separation as X and Y
df['board_disposition'] = df['board_disposition'].astype('category').cat.codes
df['complainant_ethnicity'] = df['complainant_ethnicity'].astype('category').cat.codes
df['fado_type'] = df['fado_type'].astype('category').cat.codes
# X = np.asarray(df[["year_received", "year_closed", "rank_incident", "mos_ethnicity", "mos_gender", "mos_age_incident", "complainant_ethnicity", "complainant_gender", "complainant_age_incident", "allegation", "contact_reason", "outcome_description", "board_disposition"]])
X = np.asarray(df[['board_disposition', 'complainant_ethnicity']])
y = np.asarray(df['fado_type'])

# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Model Building - Linear Regression

# Training the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Applying the model to make a prediciton
y_lr_train_pred = lr.predict(X_train) # making prediction on the original dataset it was trained on
y_lr_test_pred = lr.predict(X_test)
print(y_lr_train_pred, y_lr_test_pred)

# Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print('LR MSE (Train): ', lr_train_mse)
# print('LR R2 (Train): ', lr_train_r2)
# print('LR MSE (Test): ', lr_test_mse)
# print('LR R2 (Test): ', lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

from tabulate import tabulate
table = [['LR MSE (Train)', lr_train_mse], ['LR R2 (Train)', lr_train_r2],
         ['LR MSE (Test)', lr_test_mse], ['LR R2 (Test)', lr_test_r2]]
print(tabulate(table))

# import matplotlib.pyplot as plt
# # plt.plot([lr_test_mse], [lr_test_r2])
# plt.plot([lr_test_mse])
# plt.plot([lr_test_r2])
# plt.xlabel('LR MSE (Test)')
# plt.ylabel('LR R2 (Test)')
# plt.show()

