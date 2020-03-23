import numpy as np
from Neuron import Neuron


class Layer:

    def __init__(self, number_of_neurons, whichLayer):
        self.number_of_neurons = number_of_neurons
        self.neuronsTab = []
        self.whichLayer = whichLayer

    def make_layer(self, numberOfInputs, numberOfOutputs):
        for j in range(self.number_of_neurons):
            self.neuronsTab.append(
                Neuron(numberOfInputs, numberOfOutputs, j+1)
            )

    def __str__(self):
        return f"Layer number: {self.whichLayer} \n" \
               f"Number of neurons: {self.number_of_neurons}\n" \
               f"{self.neuronsTab}"
