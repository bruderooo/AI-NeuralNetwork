import numpy as np


class Neuron:

    # 'inputData' - it is array with input data
    # 'bias' - it have True or False
    # 'outputData' - it is array with output data
    def __init__(self, inputData, bias, outputData):
        self.inputData = np.array(inputData)
        self.bias = bias
        self.outputData = outputData
        self.weights = np.array([])
        self.sigma = 0.0

    def makeWeights(self):
        self.weights = 2 * np.random.rand(len(self.inputData), len(self.outputData)) - 1

    def train(self, epochs):
        if self.bias:
            self.bias = 0.0
            self.bias = 2 * np.random.rand(len(self.inputData), 1) - 1
        else:
            self.bias = 0.0

        for epoch in range(epochs):
            sigma = sum(self.inputData * self.weights) + self.bias
            self.__activationFunction(sigma)

    def __activationFunction(self, x):
        e = np.exp(x)
        return e / (e + 1)

    def __activationFunctionDerivative(self, x):
        fun = self.__activationFunction(x)
        return fun * (1 - fun)


neuron = Neuron(2, True, 2)
