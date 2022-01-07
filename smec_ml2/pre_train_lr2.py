# TODO: 虽然是分开了，但是效果不好，说明单纯的Linear还是不行

from sklearn.linear_model import LinearRegression
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error


def read_data_from_txt(datafile, feature_i):
    Xs = []
    ys = []
    with open(datafile) as f:
        for line in f:
            if line == '' or line == '\n':
                continue
            # line = line.replace('\'', '\"')
            # print(line)
            data = json.loads(line)
            x = data['x_prime']
            y = data['y']
            if feature_i != -1 and y[feature_i] == 0:
                continue

            Xs.append(x)
            if feature_i == -1:
                ys.append(y)
            else:
                ys.append(y[feature_i])
    return np.array(Xs), np.array(ys)


model = [LinearRegression() for i in range(32)]
# DATASET = './dataset16_mini.txt'
DATASET = './dataset16.txt'
for i in range(32):
    X, y = read_data_from_txt(DATASET, i)
    if X.shape[0] != 0:
        model[i].fit(X, y)
        y_predict_i = model[i].predict(X)
        mse = mean_squared_error(y_predict_i, y)
        print("Diff: {} {}".format(np.sqrt(mse), mse))

# import pickle
# with open('clf.pickle', 'wb') as f:
#     pickle.dump(model, f)

# np.set_printoptions(precision=3, suppress=True, linewidth=300)
#     a = input('')
