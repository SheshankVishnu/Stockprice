import pandas as pd
import numpy as np
import matplotlib as plt

data = pd.read_csv("TSLA.csv")

X  = data.iloc[:,3].values

print(np.shape(X))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(X)