import numpy as np


class Neuron:

    def __init__(self, numberOfInputs, numberOfOutputs, neuronIndex):
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.weights = 2 * np.random.sample(self.numberOfInputs) - 1
        self.neuronIndex = neuronIndex

    def sum(self, x):
        y = np.dot(x, self.weights)
        return y

    def __str__(self):
        return f"\nNeuron {self.neuronIndex} \n" \
               f"Number of inputs: {self.numberOfInputs} \n" \
               f"Number of outputs: {self.numberOfOutputs} \n" \
               f"Weights: {self.weights}"

    def __repr__(self):
        return "\n" + self.__str__()
