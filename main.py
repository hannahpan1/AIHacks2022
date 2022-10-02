import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

initial_df = pd.read_csv('allegations.csv')
initial_df = initial_df[['year_received', "month_received", 'fado_type', 'precinct', 'board_disposition']]

df = initial_df[(initial_df["fado_type"] == "Abuse of Authority")]
df["date"] = pd.to_datetime(df.year_received.astype(str) + '/' + df.month_received.astype(str) + '/01')
df = df.groupby(['date']).size().to_frame('fado_size').reset_index()
df.index = df['date']
df = df.sort_index(ascending=True)

data = pd.DataFrame(index=range(0, len(df)), columns=['date', 'fado_size'])
for i in range(0, len(data)):
    data["date"][i] = df['date'][i]
    data["fado_size"][i] = df["fado_size"][i]

scaler= MinMaxScaler(feature_range=(0,1))
data.index = data.date
data.drop("date",axis=1,inplace=True)

final_data = data.values
train_data = final_data[0:175, :]
valid_data = final_data[175:, :]
scaled_data = scaler.fit_transform(final_data)
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])


lstm_model = Sequential(
    [LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train)[1], 1)),
     LSTM(units=50),
     Dense(1)]
)


model_data = data[len(data) - len(valid_data) - 60:].values
model_data = model_data.reshape(-1, 1)
model_data = scaler.transform(model_data)

lstm_model.compile(loss='mean_squared_error', optimizer ='adam')
lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
X_test = []
for i in range(60, model_data.shape[0]):
    X_test.append(model_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_fado_num = lstm_model.predict(X_test)
predicted_fado_num = scaler.inverse_transform(predicted_fado_num)

train_data = data[:175]
valid_data = data[175:]
valid_data['Predictions'] = predicted_fado_num
plt.plot(train_data["fado_size"])
plt.plot(valid_data[['fado_size', "Predictions"]])