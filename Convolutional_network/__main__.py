import os
import sys
import math
from datetime import datetime

from Convolutional_network import Convolutional_network as CN

sys.path.insert(0, './RNA')
from Perceptron import Perceptron as Classifier

import imageio.v3 as iio
import matplotlib.pyplot as plt

# File configuration
Execution_mode = 2                                                          # 1 for training, 2 to charge weights
file_name = "./Convolutional_network/Weights_datasets/Perceptron-1-25_04_2024-09_56_46.txt"   # File name to load the weights
Save_weights = False                                                         # True to save the weights in a file (set the file name in the exercises section)
Exercise_id = 1                                                             
debug = 1                                                                   # 0 for no debug, 1 for debug cost function, 2 for debug weights

# Load images dataset
images_folder = "./training_resouces/characters/"
images_path_list = os.listdir(images_folder)

# Convolution network structure
filters = [
            [[0, 1, 0],
             [0, 1, 0],
             [0, 1, 0]],
            [[0, 0, 0],
             [1, 1, 1],
             [0, 0, 0]],
            [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
           ]
network = CN.Convolutional_network(1, filters, filtering_function="ReLU", compresion_rule="ponderation", compression_rate=2)

# Generate dataset
training_list = []
for image_path in images_path_list:
    image_array = iio.imread(images_folder+image_path)

    image_item_input = []
    im_size = image_array.shape
    image_item_input = [[0 for i in range(im_size[1])] for j in range(im_size[0])]

    for i in range(im_size[0]):
        for j in range(im_size[1]):
            if image_array[i][j]:
                image_item_input[i][j] = 0
            else:
                image_item_input[i][j] = 1

    network_output = network.Calculate_output([image_item_input])

    training_item_input = []

    for image_pack in network_output:
        for image in image_pack:
            for row in image:
                for col in row:
                    training_item_input.append(col)

    training_list_output = [0,0,0,0,0,0]
    if image_path[3][0] == 'a':
        training_list_output = [1,0,0,0,0,0]
    elif image_path[3][0] == 'b':
        training_list_output = [0,1,0,0,0,0]
    elif image_path[3][0] == 'c':
        training_list_output = [0,0,1,0,0,0]
    elif image_path[3][0] == 'd':
        training_list_output = [0,0,0,1,0,0]
    elif image_path[3][0] == 'e':
        training_list_output = [0,0,0,0,1,0]
    elif image_path[3][0] == 'f':
        training_list_output = [0,0,0,0,0,1]
           
    training_list.append([training_item_input, training_list_output])

# Split dataset in training and comprobation
comprobation_list = training_list[55:]
training_list = training_list[:55]

# Create clasifier and to train it with the dataset
RNA_structure = [40, 15, len(training_list[1][1])]
classifier = Classifier(len(training_list[1][0]), RNA_structure, 'sigmoid',learning_rate=0.1, lambda_= 25*math.pow(10,-8))

# Show initial hipotesis of the classifier
for training_item in comprobation_list:
    print("Initial hipotesis for case Y = ", training_item[1] ,"] is ", classifier.Calculate_output(training_item[0]) )

# Execute the classifier mode
if Execution_mode == 1:
    J_evolution = classifier.Training(training_list, iterations=5000, convergence_magnitud=5, debug=debug)
else:
    classifier.Set_weights(file_name)

print("\n ---- RNA calibrated ---- \n")

# Show final hipotesis of the classifier
for training_item in comprobation_list:
    print("Final hipotesis for case Y = ", training_item[1] ,"] is ", classifier.Calculate_output(training_item[0]) )

# Save weights in a file
if Save_weights:
    file_name = "./CNN/Weights_datasets/Perceptron-"+ str(Exercise_id) +"-" + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".txt"
    classifier.Export_weights_to_file(file_name)
    print("Weights saved in ", file_name)


exit() # comment this line to show the images generated by CNN
first_image = comprobation_list[0]
fig, axs = plt.subplots(len(first_image))
fig.show()

i = 0
for image in first_image[0]:
    plt.ion()
    result = axs[i].imshow(image)
    i += 1

while True:
    plt.pause(0.05)
    pass
