from datetime import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from database.config import connection
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

register_matplotlib_converters()
from pathlib import Path
from time import time

# creating database connection
mydb = connection()
cursor = mydb.cursor()

#inisialization data
data = []

# read data
visitor_data = pd.read_sql("SELECT date, total_visitor FROM customer_data_dummy", mydb, index_col='date', parse_dates={"date": {"format": "%Y-%m-%d"}})
# visitor_data = pd.read_excel(Path('./jumlah_pengunjung.xlsx'), sheet_name="jumlah_pengunjung", index_col=0)

#infer the frequency of the data
visitor_data = visitor_data.asfreq(pd.infer_freq(visitor_data.index))
length_data = visitor_data.shape[0]
train_length = round(length_data * 0.9)
plt.figure(figsize=(10, 4))
plt.plot(visitor_data)
plt.title('Total Visitor', fontsize=20)
plt.ylabel('visitors')
plt.show()

## Remove the trend
# differencing the data to make the data stationer
first_diff = visitor_data.diff()[1:]
plt.figure(figsize=(10, 4))
plt.plot(first_diff)
plt.title('The Stationary of Total Visitor Data', fontsize=20)
plt.ylabel('visitors')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.show()

fig, ax = plt.subplots(figsize=(8, 3))
plot_acf(first_diff, ax=ax, lags=30)
plt.show()
## => Based on ACF, we should start with a seasonal MA process

# PACF
fig, ax = plt.subplots(figsize=(8, 3))
plot_pacf(first_diff, ax=ax, lags=30)
plt.show()
## => Based on PACF, we should start with a seasonal AR process

train_data = visitor_data[:train_length]
test_data = visitor_data[train_length+1:]

## Fit the SARIMA Model
# define model
my_order = (0, 1, 0)
my_seasonal_order = (1, 0, 1, 12)
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

# fit the model
start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

# summary of the model
print(model_fit.summary())

#get the predictions and residuals
predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)

residuals = test_data['total_visitor'] - predictions

#plot the residuals
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.axhline(0, linestyle='--', color='k', alpha=0.5)
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error')
plt.show()

#plot the predictions
plt.figure(figsize=(10, 4))
plt.plot(visitor_data)
plt.plot(predictions)
plt.legend(('Data', 'Predictions'))
plt.title('Total Visitor Dataset', fontsize=20)
plt.ylabel('Visitors')
plt.show()

#get the mean error of the model
mean_error = np.mean(abs(residuals/test_data['total_visitor']))*100 
print('Mean Absolute Percent Error:', round(mean_error, 2), "%")

#get the root mean error of the model
root_mean = np.sqrt(np.mean(residuals**2))
print('Root Mean Squared Error:', round(root_mean, 2), '%')

# import pickle
# filename = 'c:/Users/user/Desktop/model_sarima.pkl'
# with open(filename, 'wb') as f:
#   loaded_model = pickle.dump(model_fit, f)
# # Test The SARIMA Model Using Empty Data

# Load the SARIMA model

# with open(filename, 'rb') as f:
#   loaded_model = pickle.load(f)

def generate_datetime(start_date):
    """ Generate dummy datetime.datetime object """
    return [start_date + pd.to_timedelta(i, unit='D') for i in range(31)]
    
# get the last data from train data
last_date = visitor_data.iloc[-1:].index.values[0]
new_date = generate_datetime(last_date)

# get the predictions and residuals
new_predictions = model_fit.forecast(len(new_date))
print(new_predictions)

for date, visitor in zip(new_date, new_predictions):
  cursor.execute(f"INSERT INTO visitor_pred_res (date, visitor) VALUES ('{date}', {visitor})")
  mydb.commit()  # to make final output we have to run the 'commit()' method of the database object
print(cursor.rowcount, "record inserted")

# # plot the prediction result
# plt.figure(figsize=(10, 4))
# plt.plot(df_test)
# plt.plot(new_predictions)
# plt.title('Predictions Data Visitors', fontsize=20)
# plt.ylabel('Visitors')
# plt.xticks(rotation=45)
# plt.show()

mydb.close()
