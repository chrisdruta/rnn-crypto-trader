#!/usr/bin/env python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert series to supervised learning
def prepareData(data, n_in=1, n_out=1):
	"""
	Given data is processed and bundled along the time axis according to delay amounts n_in and n_out

	Args:
		data: data to prepare and split
		n_in: Amount of lag days in
		n_out: Amount of lag days out
	
	Returns:
		List of data with same length as input data but prepared for LSTM
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# Input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-1 * i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
	reframed = pd.concat(cols, axis=1)
	reframed.columns = names
	reframed.dropna(inplace=True)

	return reframed

def up_down(stuff):
    binary = []
    for i in range(1, len(stuff)):
        if stuff[i] > stuff[i-1]:
            binary.append(1)
        elif stuff[i] < stuff[i-1]:
            binary.append(-1)
        else:
            binary.append(0)
    return binary

def compare_updown(first, second):
    correct = 0
    total = len(first)
    for i in range(len(first)):
        if first[i] == second[i]:
            correct += 1
    return correct, correct / total

# Load dataset
dataset = pd.read_csv("./data/bitcoin_parsed.csv", header=None, index_col=0)
data = dataset.values.astype('float32')
print(f"Length of data: {len(data)}")

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

# Preparing data for LSTM layer input
nDays = 1; nFeatures = 5
data = prepareData(scaled, nDays, nDays).values

# Split into training and testing sets
#trainPercent = 0.975
trainPercent = 0.90
pivot = int(len(data) * trainPercent)
train = data[:pivot, :]
test = data[pivot:, :]

# Split training and testing sets into input and outputs for fitting
train_X, train_y = train[:, :nDays * nFeatures], train[:, -1 * nFeatures]
test_X, test_y = test[:, :nDays * nFeatures], test[:, -1 * nFeatures]

# Reshape to [samples x time steps x features]
train_X = train_X.reshape((train_X.shape[0], nDays, nFeatures))
test_X = test_X.reshape((test_X.shape[0], nDays, nFeatures))

model = keras.models.Sequential([
    keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])),
    keras.layers.Dense(1)
])
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)

# Make future predictions
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], nDays * nFeatures))

# Invert data scale
inv_yhat = np.concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Calculate RMSE
rmse = mean_squared_error(inv_y, inv_yhat)**0.5

# Calculate binary increase to decrease ratio
real, pred = up_down(inv_y), up_down(inv_yhat)
dirPredRate = compare_updown(real, pred)[1]

# Generate and save graph
plt.figure()
plt.title(f"RMSE: {round(rmse, 3)}, Direction Prediction Rate: {round(dirPredRate, 3)}")
plt.plot(range(len(inv_y)), inv_y, 'b')
plt.plot(range(len(inv_yhat)), inv_yhat, 'r')

plt.xlabel('Time (Days)')
plt.ylabel('BTC Price ($)')
plt.legend(['Real', 'Prediction'])

plt.show()
#plt.savefig("./output/{}.png".format(filename), bbox_inches='tight', dpi=400)
quit()

# Run hypothetical investment scenarios and write results to file
most_stocks, most_money = stock_game(inv_y, real, 10000)
predicted_stocks, predicted_money = stock_game(inv_y, pred, 10000)
with open("./output/{}.txt".format(filename), "w") as f:
    f.write("Possible stocks and money: %.2f\t$%.2f\n" % (most_stocks, most_money))
    f.write("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))
print("Total stocks and money: %.2f\t$%.2f" % (most_stocks, most_money))
print("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))
