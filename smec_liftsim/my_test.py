import numpy as np

a = np.ones((3,2,2)) * 0.5
b = np.random.random((2,3,2,2))

print(b)

c = a < b
print(c)