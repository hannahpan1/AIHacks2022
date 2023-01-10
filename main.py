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


#Tentative plan at neural networks to predict future complaints over time.

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline
# from matplotlib.pylab import rcParams

# rcParams['figure.figsize'] = 20, 10
# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense
# from sklearn.preprocessing import MinMaxScaler

# initial_df = pd.read_csv('allegations.csv')
# initial_df = initial_df[['year_received', "month_received", 'fado_type', 'precinct', 'board_disposition']]

# df = initial_df[(initial_df["fado_type"] == "Abuse of Authority")]
# df["date"] = pd.to_datetime(df.year_received.astype(str) + '/' + df.month_received.astype(str) + '/01')
# df = df.groupby(['date']).size().to_frame('fado_size').reset_index()
# df.index = df['date']
# df = df.sort_index(ascending=True)

# data = pd.DataFrame(index=range(0, len(df)), columns=['date', 'fado_size'])
# for i in range(0, len(data)):
#     data["date"][i] = df['date'][i]
#     data["fado_size"][i] = df["fado_size"][i]

# scaler= MinMaxScaler(feature_range=(0,1))
# data.index = data.date
# data.drop("date",axis=1,inplace=True)

# final_data = data.values
# train_data = final_data[0:175, :]
# valid_data = final_data[175:, :]
# scaled_data = scaler.fit_transform(final_data)
# x_train, y_train = [], []
# for i in range(60, len(train_data)):
#     x_train.append(scaled_data[i - 60:i, 0])
#     y_train.append(scaled_data[i, 0])


# lstm_model = Sequential(
#     [LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train)[1], 1)),
#      LSTM(units=50),
#      Dense(1)]
# )


# model_data = data[len(data) - len(valid_data) - 60:].values
# model_data = model_data.reshape(-1, 1)
# model_data = scaler.transform(model_data)

# lstm_model.compile(loss='mean_squared_error', optimizer ='adam')
# lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
# X_test = []
# for i in range(60, model_data.shape[0]):
#     X_test.append(model_data[i - 60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predicted_fado_num = lstm_model.predict(X_test)
# predicted_fado_num = scaler.inverse_transform(predicted_fado_num)

# train_data = data[:175]
# valid_data = data[175:]
# valid_data['Predictions'] = predicted_fado_num
# plt.plot(train_data["fado_size"])
# plt.plot(valid_data[['fado_size', "Predictions"]])
