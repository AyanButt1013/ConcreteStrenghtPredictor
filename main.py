# Importing Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Loading Data Frame
path = 'Concrete_Data.xls'
df = pd.read_excel(path)
df_features = df.copy()
df_labels = df_features.pop('Compr. Str(MPA)')
X = np.array(df_features)
Y = np.array(df_labels)

# Normalization
normalize = tf.keras.layers.Normalization(axis=-1)
normalize.adapt(X)
Xn = normalize(X)
Xn =np.array(Xn)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(Xn, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBRegressor()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Compression Test and predicted data")
plt.legend()
plt.show()

