from sklearn.linear_model import LinearRegression
import json
from sklearn.model_selection import train_test_split
import numpy as np


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
has_data = [0 for i in range(32)]
# DATASET = './dataset16_mini.txt'
DATASET = './dataset16.txt'
for i in range(32):
    X, y = read_data_from_txt(DATASET, i)
    if X.shape[0] != 0:
        model[i].fit(X, y)
        has_data[i] = 1

# import pickle
# with open('clf.pickle', 'wb') as f:
#     pickle.dump(model, f)

from sklearn.metrics import mean_squared_error

X, y = read_data_from_txt(DATASET, -1)

y_predicts = []
for i in range(32):
    if not has_data[i]:
        y_predict_i = np.zeros(X.shape[0])
    else:
        y_predict_i = model[i].predict(X)
    # print(y_predict_i.shape)
    mask = y[:, i] > 0
    y_predict_i[1-mask] = 0
    y_predicts.append(y_predict_i)
    print(y_predict_i)
    print(y[:, i])

y_predicts = np.stack(y_predicts, axis=1)
print(y_predicts.shape)
print(y.shape)

mse = mean_squared_error(y_predicts, y)
print("Diff: {} {}".format(np.sqrt(mse), mse))
# np.set_printoptions(precision=3, suppress=True, linewidth=300)
#     a = input('')
