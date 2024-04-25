from RNA.neuron import Neuron
import math

class Neural_layer:

    # --- CONSTRUCTOR ---

    def __init__(self, num_neurons, num_inputs, activation_function):
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
    
    # --- METHODS ---
    
    # Calculate the output of the neurons of the layer
    def Output_calculation(self, inputs):
        return [neuron.Output_calculation(inputs) for neuron in self.neurons]
    
    # Return the outputs of the neurons of the layer
    def Get_outputs(self):
        return [neuron.output for neuron in self.neurons]
    
    # Calculate the cost of the weights of the layer
    def Calculate_weights_costs(self):
        cost = 0
    
        for neuron in self.neurons:
            weight_list = [math.pow(weight,2) for weight in neuron.weights]
            for weight in weight_list:
                cost += weight
            cost += math.pow(neuron.bias,2)

        return cost
    
    # Modify the weights of the neurons of the layer considering learning rate
    def Step_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.Step_weights(learning_rate)

    # Clean the cost gradients of the neurons of the layer
    def Clean_cost_gradients(self):
        for neuron in self.neurons:
            neuron.Clean_cost_gradients()

    # Return the weights of the neurons of the layer
    def Get_weights(self):
        return [neuron.Get_weights() for neuron in self.neurons]
    
    # Configure the weights of the neurons of the layer
    def Set_weights(self, layer_weights):
        for i in range(len(self.neurons)):
            self.neurons[i].Set_weights(layer_weights[i])
