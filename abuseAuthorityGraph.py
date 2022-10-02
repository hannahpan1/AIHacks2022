import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import datetime
import plotly.express as px
import plotly.graph_objects as go

initial_df = pd.read_csv('allegations.csv')
initial_df["date"] = pd.to_datetime(initial_df.year_received.astype(str) + '/' + initial_df.month_received.astype(str) + '/01')
initial_df['date'] = initial_df['date'].map(datetime.datetime.toordinal)

df1 = initial_df[(initial_df["fado_type"] == "Abuse of Authority")]
df1 = df1.groupby(['date']).size().to_frame('fado_size').reset_index()
X = np.asarray(df1[['date']])
y = np.asarray(df1['fado_size'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Model Building - Linear Regression

# Training the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Applying the model to make a prediciton
y_lr_train_pred = lr.predict(X_train) # making prediction on the original dataset it was trained on
y_lr_test_pred = lr.predict(X_test)

df = px.data.tips()
fig = px.scatter(x=np.concatenate(X_test), y=y_test)
fig.add_trace(go.Scatter(x=np.concatenate(X_test), y=y_lr_test_pred, line=dict(color="#011269")))

fig.update_layout(
    title = 'Abuse of Authority Complaints over Time', # adding the title
    xaxis_title = 'Dates (year & month)', # title for x axis
    yaxis_title = 'Complaints', # title for y axis
    xaxis = dict(           # attribures for x axis
        showline = True,
        showgrid = True,
        linecolor = 'black',
        tickfont = dict(
            family = 'Calibri'
        )
    ),
    yaxis = dict(           # attribures for y axis
        showline = True,
        showgrid = True,
        linecolor = 'black',
        tickfont = dict(
            family = 'Times New Roman'
        )
    ),
    plot_bgcolor = 'white'  # background color for the graph
)

fig.show()


