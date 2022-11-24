import numpy as np
import math
import random
import pickle
from enum import Enum
import logging


logger = logging.getLogger('root')

class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Network:
    def __init__(self, layers_count):
        if len(layers_count) >= 3:
            self.output = None
            self.node_difference = None
            self.layers_count = layers_count
            self.layers = np.empty(shape=(len(layers_count)), dtype=np.object0)
            self.layers[0] = Layer(LayerType.INPUT,layers_count[0], None)
            for num in range(1, len(layers_count)):
                self.layers[num] = Layer(LayerType.HIDDEN,layers_count[num], self.layers[num-1])


    def save(self,num):
        pickle.dump(self, open(f"dumps/network{num}.dump", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(num):
        return pickle.load(open(f"dumps/network{num}.dump", 'rb'))

    def flush(self):
        for i, layer in np.ndenumerate(self.layers):
            layer.flush()

    def calculate_cost(self, number):
        if self.output is None:
            return
        true_values = np.zeros(shape=(self.layers_count[-1])) 
        true_values[number] = 1
        self.node_difference = true_values - self.output
        cost_per_output_node = np.square(self.node_difference)
        return np.average(cost_per_output_node)

    def feed_forward(self, image):
        self.layers[0].populate_inputs(image)
        self.layers[-1].calculate()
        self.output = np.asarray(self.layers[-1].get_values())
        return self.output

    def backpropagate(self):
        a_ratio = np.multiply(self.node_difference,2)
        for i in range(len(self.layers_count)-1,0,-1):
            a_ratio = self.layers[i].propogate_layer(a_ratio)
    
    def nudge(self):
        for i in range(1,len(self.layers_count)):
            self.layers[i].nudge_nodes()


class Layer:
    def __init__(self, layer_type: LayerType, neurons: int, previous_layer):
        self.nodes = np.empty(shape=(neurons), dtype=np.object0)
        self.layer_type = layer_type
        self.previous_layer = previous_layer
        for l in range(0, neurons):
            self.nodes[l] = Node(layer_type=layer_type, previous_layer=previous_layer)

    def flush(self):
        for i, node in np.ndenumerate(self.nodes):
            node.value = None
            
    def nudge_nodes(self):
        for i, node in np.ndenumerate(self.nodes):
            self.nodes[i].nudge_params()

    def calculate(self):
        for i,node in np.ndenumerate(self.nodes):
            node.calculate_outcome()
        
    def populate_inputs(self, image):
        if self.layer_type is LayerType.INPUT:
            for i,node in np.ndenumerate(self.nodes):
                value = int.from_bytes(image[i], "big") / 255
                node.value = value
                
    def count(self):
        return len(self.nodes)
    
    def get_value(self,index):
        if self.nodes[index].value is None:
            return self.nodes[index].calculate_outcome()
        return self.nodes[index].value
    
    def get_values(self):
        return list([node.value for i,node in np.ndenumerate(self.nodes)])
            
    def propogate_layer(self, a_ratios):
        all_activation_ratios = []
        for i,node in np.ndenumerate(self.nodes):
            all_activation_ratios.append(node.calculate_derivatives(a_ratios[i]))
        all_activation_ratios = np.asarray(all_activation_ratios)
        return np.average(all_activation_ratios, axis=0)


class Node:
    def __init__(self, layer_type: LayerType, previous_layer: Layer):
        self.value = None
        self.layer_type = layer_type
        self.previous_layer = previous_layer
        if layer_type != LayerType.INPUT:
            self.weights = np.array([None] * previous_layer.count())
            self.weight_nudges = [[] for i in range(previous_layer.count())]
            self.bias_nudges = []
            self.bias = None
            self.generate_parameters()

    def generate_parameters(self):
        if self.layer_type != LayerType.INPUT:
            for i in range(0, len(self.weights)):
                self.weights[i] = (random.random() * 2) - 1
            self.bias = (random.random() * 2) - 1

    def calculate_outcome(self):
        if self.layer_type != LayerType.INPUT:
            node_sum = self.bias
            for i in range(0, self.previous_layer.count()):
                node_sum += self.previous_layer.get_value(i) * self.weights[i]
            self.value = sigmoid(node_sum)
            return node_sum
        else: return self.value
    
    def calculate_derivatives(self, a_ratio):
        z = sigmoid(a_ratio) * (1 - sigmoid(a_ratio))
        self.bias_nudges.append(z)
        for i,weight in enumerate(self.weight_nudges):
            weight.append(self.previous_layer.get_value(i)*z)
        return [weight*z for weight in self.weights]
        
    def nudge_params(self):
        np_weight_nudges = np.asarray(self.weight_nudges)
        gradient_weight = np.multiply(np.mean(np_weight_nudges, axis=1),0.001)
        self.weights = self.weights - gradient_weight
        gradient_bias = 0.001 * np.mean(self.bias_nudges)
        self.bias -= gradient_bias

        #self.weights = np.average(np_weight_nudges, axis=0)
        #self.bias = self.bias - sum(self.bias_nudges)
        self.bias_nudges = []
        self.weight_nudges = [[] for i in range(self.previous_layer.count())]



# static function required:
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig
