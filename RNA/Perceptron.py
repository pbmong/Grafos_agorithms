from RNA.layer import Neural_layer

import math

class Perceptron:

    # --- CONSTUCTOR ---

    def __init__(self, input_size, layers_config, activation_function, learning_rate = 0.1, lambda_ = 0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.layers = []
        self.lambda_ = lambda_
        
        for i in range(len(layers_config)):
            if i == 0:
                self.layers.append(Neural_layer(layers_config[i], input_size, activation_function))
            else:
                self.layers.append(Neural_layer(layers_config[i], layers_config[i - 1], activation_function))

    # --- PUBLIC METHODS ---

    # Train the Perceptron with an specific training list
    def Training(self, training_list, iterations = 10000, convergence_magnitud = 5, debug = 0):
        past_cost = math.inf
        J_evolution = []

        for i in range(iterations):
            cost = self.__Calculate_cost(training_list)
            if past_cost - cost < math.pow(10,-convergence_magnitud):
                print(" - Converged in iteration: ", i, " with cost: ", cost)
                break
            else:
                past_cost = cost

            self.__Calculate_cost_gradient_per_weight(training_list)
            self.__Step_weights()
            self.__Clean_cost_gradients()
        
            if debug > 0:
                print("Iter: ", i, " Cost: ", cost)
                J_evolution.append(cost)
                if debug > 1:
                    weight_list = self.Get_weights()
                    for i in range(len(weight_list)):
                        print("Layer ", i, ":")
                        for j in range(len(weight_list[i])):
                            print("Neuron ",j, ":", weight_list[i][j])
                    print("\n----------------------\n")

    # Calculate the output of the Perceptron
    def Calculate_output(self, inputs):
        output = []
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].Output_calculation(inputs)
            else:
                self.layers[i].Output_calculation(self.layers[i - 1].Get_outputs())

        return self.layers[len(self.layers)-1].Get_outputs()

    # Get the weights of the Perceptron
    def Get_weights(self):
        return [layer.Get_weights() for layer in self.layers]
    
    # Show the weights of the Perceptron in the console
    def Print_weights(self):
        print("------ RNA weights -----")
        weight_list = self.Get_weights()
        for i in range(len(weight_list)):
            print("Layer ", i, ":")
            for j in range(len(weight_list[i])):
                print("Neuron ",j, ":", weight_list[i][j])
            print("----------------------")

    # Set the weights of the Perceptron from a file
    def Set_weights(self, weights_file_name):
        weights_file = open(weights_file_name,"r")
        weights_str = weights_file.read()
        weights_file.close()

        Perceptron_weights_array = self.__Parse_weights(weights_str)

        for i in range(len(self.layers)):
            self.layers[i].Set_weights(Perceptron_weights_array[i])
    
    # Save the weights of the Perceptron in a file
    def Export_weights_to_file(self, file_name):
        weights_file = open(file_name,"w")
        for layer in self.Get_weights():
            weights_file.write("l\n")
            for neuron in layer:
                weights_file.write("n\n")
                weights_file.write(str(neuron))
                weights_file.write("\n")
        weights_file.close()                            

    # --- PRIVATE METHODS ---

    # Calculate the current cost function of the Perceptron
    def __Calculate_cost(self, training_list):
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
    
    # Calculate the error of the neurons of the layers
    def __Calculate_neural_errors(self, training_item):
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

    # Calculate the cost gradients of the weights of the layers
    def __Calculate_cost_gradient_per_weight(self, training_list):
        
        for training_item in training_list:
            self.Calculate_output(training_item[0])
            self.__Calculate_neural_errors(training_item)

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

    # Update the weights of the layers
    def __Step_weights(self):
        for layer in self.layers:
            layer.Step_weights(self.learning_rate)
    
    # Clean the cost gradients of the layers
    def __Clean_cost_gradients(self):
        for layer in self.layers:
            layer.Clean_cost_gradients()

    def __Parse_weights(self, weights_str):
        Perceptron_weights_array = []

        # Extract layers of the file
        layers = weights_str.split("l\n")
        for layer in layers[1:len(layers)]:

            layer_weights_array = []
            neurons = layer.split("n\n")

            # Extract neurons of the layer of the file
            for neuron in neurons[1:]:
                weights = []
                neuron = neuron.replace('[', '')
                neuron = neuron.replace(']', '')
                neuron = neuron.replace('\n', '')

                weights_str = (neuron.split(","))

                # Extract bias parameter
                bias = float(weights_str[0])

                # Extract weights parameters
                for weight in weights_str[1:]:
                    if weight != "":
                        weights.append(float(weight))

                # Add bias and weights to the layer
                layer_weights_array.append([bias, weights])
            
            # Add layer to the Perceptron
            Perceptron_weights_array.append(layer_weights_array)

        return Perceptron_weights_array
