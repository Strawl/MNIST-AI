import numpy as np
import math
import random


class Network:

    def __init__(self, layers_count):
        if len(layers_count) >= 3:
            self.output = []
            self.layers = []
            self.correct = 0
            for num in range(0, len(layers_count)):
                if num == 0:
                    array = np.empty(shape=(layers_count[num]), dtype=np.object0)
                    for l in range(0, layers_count[num]):
                        array[l] = Node(0, False)
                    self.layers.append(array)
                else:
                    array = np.empty(shape=(layers_count[num]), dtype=np.object0)
                    for l in range(0, layers_count[num]):
                        array[l] = Node(layers_count[num - 1], True)
                    self.layers.append(array)

    def calculate_cost(self, data, number, itype):
        if len(self.output) == 0:
            self.feed_forward(data=data, number=number, itype=itype)
        should_be_value = data.get_label(itype=itype, number=number, opt=True)
        right_values = [0]*10
        right_values[should_be_value] = 1
        cost = 0
        for val in range(0, len(right_values)):
            cost += (self.output[val]-right_values[val])**2
        if should_be_value == self.output.index(max(self.output)):
            self.correct += 1
        return cost

    def feed_forward(self, data, number, itype):
        image = data.get_image(itype=itype, number=number, opt=True)
        for node in range(0, len(self.layers[0])):
            value = int.from_bytes(image[node], "big") / 255
            self.layers[0][node].set_input(value)
        for layer in range(1, len(self.layers)):
            for node in self.layers[layer]:
                node.calculate_outcome(self.layers[layer - 1])
        self.output = []
        for node in self.layers[-1]:
            self.output.append(node.value)
        return self.output


class Node:

    def __init__(self, prev_count, gener_param):
        self.prev_count = prev_count
        self.value = None
        if prev_count != 0:
            self.weights = np.array([None] * prev_count)
            self.bias = None
            if gener_param:
                self.generate_parameters()

    def generate_parameters(self):
        if self.prev_count != 0:
            for i in range(0, len(self.weights)):
                self.weights[i] = (random.random() * 2) - 1
            self.bias = ((random.random() * 2) - 1) * 5

    def set_input(self, value):
        self.value = value

    def calculate_outcome(self, prev_layer):
        if self.prev_count != 0:
            node_sum = self.bias
            for i in range(0, len(prev_layer)):
                node_sum += prev_layer[i].value * self.weights[i]
            self.set_input(sigmoid(node_sum))


# static function required:
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig
