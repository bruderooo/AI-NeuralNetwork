import numpy as np


class Neuron:

    def __init__(self, numberOfInputs, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.weights = 2 * np.random.sample(self.numberOfInputs) - 1

    def sum(self, x):
        y = np.dot(x, self.weights)
        return y
