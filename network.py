import numpy as np
import math
import random


class Network:

    def __init__(self, layers_count):
        if len(layers_count) >= 3:
            self.layers = []
            for num in range(0, len(layers_count)):
                if num == 0:
                    array = np.empty(shape=(layers_count[num]),dtype=np.object0)
                    for l in range(0, layers_count[num]):
                        array[l] = Node(0, False)
                    self.layers.append(array)
                else:
                    array = np.empty(shape=(layers_count[num]),dtype=np.object0)
                    for l in range(0, layers_count[num]):
                        array[l] = Node(layers_count[num - 1], True)
                    self.layers.append(array)

    def feed_forward(self, data, number):
        image = data.get_image(itype="train", number=number, opt=True)
        for node in range(0, len(self.layers[0])):
            value = int.from_bytes(image[node], "big") / 255
            self.layers[0][node].set_input(value)
        for layer in range(1, len(self.layers)):
            for node in self.layers[layer]:
                node.calculate_outcome(self.layers[layer - 1])
        for node in self.layers[-1]:
            print(node.value)


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
            self.bias = ((random.random() * 2) - 1) * 3

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
