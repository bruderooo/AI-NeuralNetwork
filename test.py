import numpy as np
from Layer import Layer

inp = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

out = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

test_input = np.array([1, 0, 0, 0])

layer = Layer(2, 1)
layer.make_layer(4, 4)
print(layer)
