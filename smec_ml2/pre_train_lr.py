from sklearn.linear_model import LinearRegression
import json
from sklearn.model_selection import train_test_split
import numpy as np


def read_data_from_txt(datafile):
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
            Xs.append(x)
            ys.append(y)
    return np.array(Xs), np.array(ys)


X, y = read_data_from_txt('./dataset16.txt')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

predict_value = model.predict(X_test)

import pickle
with open('clf.pickle', 'wb') as f:
    pickle.dump(model, f)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predict_value)
print("Diff: {} {}".format(np.sqrt(mse), mse))

print(X_test.shape)
print(model.coef_)
print(model.intercept_)

for i in range(X_test.shape[0]):
    x = X_test[i]
    pv = model.predict(x.reshape((1, -1)))
    print('predict: ', pv)
    print('real: ', y_test[i])
    a = input('')