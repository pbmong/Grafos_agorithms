from neuron import Neuron
import math

class Neural_layer:
    def __init__(self, num_neurons, num_inputs, activation_function):
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
    
    def Output_calculation(self, inputs):
        return [neuron.Output_calculation(inputs) for neuron in self.neurons]
    
    def Get_outputs(self):
        return [neuron.output for neuron in self.neurons]
    
    def Calculate_weights_costs(self):

        cost = 0
        for neuron in self.neurons:
            weight_list = [math.pow(weight,2) for weight in neuron.weights]
            for weight in weight_list:
                cost += weight
            cost += math.pow(neuron.bias,2)

        return cost
    
    def Step_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.Step_weights(learning_rate)

    def Clean_cost_gradients(self):
        for neuron in self.neurons:
            neuron.Clean_cost_gradients()

    def Get_weights(self):
        return [neuron.Get_weights() for neuron in self.neurons]
