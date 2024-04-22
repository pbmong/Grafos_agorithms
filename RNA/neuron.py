import random
import math

# activation functions options

activation_functions = {"sigmoid", "tanh", "relu", "leaky_relu", "linear"}

class Neuron:  

    def __init__(self, num_inputs, activation_function):
        self.bias = random.uniform(-0.5, 0.5)
        self.output = None
        self.error = 0
        self.deltas = []
        self.activation_function = activation_function

        self.weights = []
        for i in range(num_inputs):
            self.weights.append(random.uniform(-0.5, 0.5))

        self.cost_gradients = [0]*num_inputs
        self.cost_bias = 0

    def Output_calculation(self, inputs):
        
        SUM = 0
        SUM += self.bias  # bias parameter

        for i in range(len(self.weights)): # sum of the inputs multiplied by the weights
            SUM += inputs[i] * self.weights[i]

        if self.activation_function == "sigmoid": # activation function
            result = 1 / (1 + math.exp(-SUM))
        else:
            result = None
        self.output = result

        return result
    
    def Calculate_error(self, target):
        self.error = self.output - target
    
    def Clean_cost_gradients(self):
        self.cost_gradients = [0]*len(self.cost_gradients)
        self.cost_bias = 0

    def Step_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.cost_gradients[i]
        self.bias -= learning_rate * self.cost_bias

    def Get_weights(self):
        return [self.bias, self.weights]
        