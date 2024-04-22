from layer import Neural_layer

import math

class Perceptron:
    def __init__(self, input_size, layers_config, activation_function,learning_rate, lambda_ = 0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.layers = []
        self.lambda_ = lambda_
        
        for i in range(len(layers_config)):
            if i == 0:
                self.layers.append(Neural_layer(layers_config[i], input_size, activation_function))
            else:
                self.layers.append(Neural_layer(layers_config[i], layers_config[i - 1], activation_function))

    def Training(self, training_list, iterations = 10000, convergence_magnitud = 5, debug = False):
        past_cost = math.inf
        for i in range(iterations):
            cost = self._Calculate_cost(training_list)
            if past_cost - cost < math.pow(10,-convergence_magnitud):
                print("Converged in iteration: ", i, " with cost: ", cost)
                break
            else:
                past_cost = cost

            self._Calculate_cost_gradient_per_weight(training_list)
            self._Step_weights()
            self._Clean_cost_gradients()
        
            if debug:
                print("Epoch: ", i, " Cost: ", cost)
                weight_list = self.Get_weights()
                for i in range(len(weight_list)):
                    print("Layer ", i, ":")
                    for j in range(len(weight_list[i])):
                        print("Neuron ",j, ":", weight_list[i][j])
                print("\n----------------------\n")

    def Calculate_output(self, inputs):
        output = []
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].Output_calculation(inputs)
            else:
                self.layers[i].Output_calculation(self.layers[i - 1].Get_outputs())

        return self.layers[len(self.layers)-1].Get_outputs()

    def _Calculate_cost(self, training_list):
        cost = 0
        for training_item in training_list:
            hipothesis = self.Calculate_output(training_item[0])
            
            for i in range(len(training_item[1])):
                hipothesis_output = hipothesis[i]
                training_output = training_item[1][i]
                cost += training_output * math.log(hipothesis_output) + (1 - training_output)*math.log(1 - hipothesis_output)
        
        cost = -cost/len(training_list)

        weight_sum = 0
        for layer in self.layers:
            weight_sum += layer.Calculate_weights_costs()
        
        cost += self.lambda_ * weight_sum

        return cost
    
    def _Calculate_neural_errors(self, training_item):
        for i in range(len(self.layers)-1, -1, -1):

            if i == len(self.layers)-1: # Calculate the error of the output neurons
                for j in range(len(self.layers[i].neurons)): 
                    self.layers[i].neurons[j].Calculate_error(training_item[1][j])

            else:
                for j in range(len(self.layers[i].neurons)):
                    error = 0
                    for k in range(len(self.layers[i+1].neurons)): # Calculate the error of the neuron based on the error of the neurons in the next layer
                        error += self.layers[i+1].neurons[k].error * self.layers[i+1].neurons[k].weights[j]
                    
                    # Calculate the error of the neuron based on the derivative of the activation function
                    error = error * self.layers[i].neurons[j].output * (1 - self.layers[i].neurons[j].output)

                    self.layers[i].neurons[j].error = error

    def _Calculate_cost_gradient_per_weight(self, training_list):
        
        for training_item in training_list:
            self.Calculate_output(training_item[0])
            self._Calculate_neural_errors(training_item)

            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].neurons)):
                    for k in range(len(self.layers[i].neurons[j].weights)):
                        if i == 0:
                            self.layers[i].neurons[j].cost_gradients[k] += self.layers[i].neurons[j].error * training_item[0][k]
                        else:
                            self.layers[i].neurons[j].cost_gradients[k] += self.layers[i].neurons[j].error * self.layers[i-1].neurons[k].output

                    self.layers[i].neurons[j].cost_bias += self.layers[i].neurons[j].error * 1

        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.cost_gradients[i] = neuron.cost_gradients[i]/len(training_list) + self.lambda_ * neuron.weights[i]
                neuron.cost_bias = neuron.cost_bias/len(training_list)

    def _Step_weights(self):
        for layer in self.layers:
            layer.Step_weights(self.learning_rate)
    
    def _Clean_cost_gradients(self):
        for layer in self.layers:
            layer.Clean_cost_gradients()

    def Get_weights(self):
        return [layer.Get_weights() for layer in self.layers]
