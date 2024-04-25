
import math

# Class for Convolutional Neural Network
class Convolutional_network:

    # ---- CONSTRUCTOR ----
    def __init__(self, cnn_sizes, filters, filtering_function = "ReLU", compresion_rule = "ponderation", compression_rate = 2):
        self.cnn_sizes = cnn_sizes
        self.filters = filters
        self.filtering_function = filtering_function
        self.compression_rate = compression_rate
        self.compression_rule = compresion_rule

    # ---- PUBLIC METHODS ----

    def Calculate_output(self, input_list):
        
        result = []

        for input_image in input_list:
            # Calculate the convolution
            convolution_output = self.__Calculate_convolution(input_image)
            
            filtered_output = self.__Filtering_function(convolution_output)

            compresed_result = self.__Compresion(filtered_output)

            result.append(compresed_result)

        return result

    # ---- PRIVATE METHODS ----

    def __Calculate_convolution(self, input_image):

        input_image_size = [len(input_image), len(input_image[0])]
        
        convolution_result = []

        for filter in self.filters:
            
            filter_size = [len(filter),len(filter[0])]
            filter_result = [[0 for i in range(input_image_size[1] - filter_size[1] + 1)] for j in range(input_image_size[0] - filter_size[0] + 1)]

            for i in range(input_image_size[0] - filter_size[0] + 1):
                for j in range(input_image_size[1] - filter_size[1] + 1):
                    sub_image = [row[j:j+filter_size[1]] for row in input_image[i:i+filter_size[0]]]
                    filter_result[i][j] = self.__Convolution(sub_image, filter)
                    
            convolution_result.append(filter_result)

        return convolution_result
    
    def __Convolution(self, input_image, filter):
        sum = 0
        
        for i in range(len(input_image)):
            for j in range(len(input_image[0])):
                sum += input_image[i][j] * filter[i][j]

        return sum
    
    def __Filtering_function(self, input_image):

        result = []
        
        for image in input_image:
            compresed_image = []

            # Apply the filtering function
            # For ReLU, the function is f(x) = max(0, x)
            if self.filtering_function == "ReLU":
                for i in range(len(image)):
                    row = []
                    for j in range(len(image[0])):
                        if image[i][j] > 0:
                            row.append(image[i][j])
                        else:
                            row.append(0)
                    compresed_image.append(row)
            
            # For Sigmoid, the function is f(x) = 1 / (1 + e^(-x))
            elif self.filtering_function == "Sigmoid":
                for i in range(len(image)):
                    row = []
                    for j in range(len(image[0])):
                        row.append(1 / (1 + math.exp(-image[i][j])))
                    compresed_image.append(row)

            else:
                filtered_immage = image
            
            result.append(compresed_image)
        
        return result
    
    def __Compresion(self, input_image):
        
        result = []

        for image in input_image:
            compresed_image = []

            for i in range(0, len(image)-self.compression_rate, self.compression_rate):
                row = []
                for j in range(0, len(image[0])-self.compression_rate, self.compression_rate):
                    total = 0

                    # Apply the compression rule
                    # For ponderation, the function is f(x) = sum(x) / compression_rate^2
                    if self.compression_rule == "ponderation":
                        for k in range(self.compression_rate):
                            for l in range(self.compression_rate):
                                total += image[i+k][j+l]
                        sum = total / (self.compression_rate ** 2)

                    # For max, the function is f(x) = max(x)
                    elif self.compression_rule == "max":
                        sum = max([max(row[j:j+self.compression_rate]) for row in image[i:i+self.compression_rate]])


                    row.append(sum)
                compresed_image.append(row)
            
            result.append(compresed_image)
        
        return result