# Version: 1.0
from Perceptron import *
import math

import os
from datetime import datetime

import imageio.v3 as iio

#Select the training list and the RNA structure
Execution_mode = 2      # 1 for training, 2 to charge weights
Exercise_id = 3         # 1 for AND, 2 for AND and OR, 3 for characters
Save_weights = False     # True to save the weights in a file (set the file name in the exercises section)
Debug = 1               # 0 for no debug, 1 for debug cost function, 2 for debug weights


# Exercices

if Exercise_id == 1:
    training_list_1=[
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[1]]
    ]
    RNA_1 = [1]
    training_list = training_list_1
    RNA_structure = RNA_1
    comprobation_list = training_list_1
    
    if Execution_mode == 1:
        convergence_magnitud=4
    else:
        file_name = "./RNA/Weights_datasets/Perceptron-1.txt"

elif Exercise_id == 2:
    training_list_2=[
            [[0,0],[0,0]],
            [[0,1],[0,1]],
            [[1,0],[0,1]],
            [[1,1],[1,1]]
        ]
    RNA_2 = [3, 2]
    training_list = training_list_2
    RNA_structure = RNA_2
    comprobation_list = training_list_2
    
    if Execution_mode == 1:
        convergence_magnitud=4
    else:
        file_name = "./RNA/Weights_datasets/Perceptron-2-24_04_2024-13_20_21.txt"


elif Exercise_id == 3:
    
    images_folder = "./training_resouces/characters/"
    images_path_list = os.listdir(images_folder)
        
    training_list_3 = []
    for image_path in images_path_list:
        image_array = iio.imread(images_folder+image_path)

        training_list_input = []
        for row in image_array:
            for col in row:
                if col:
                    training_list_input.append(0)
                else:
                    training_list_input.append(1)
            
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
            
        training_list_3.append([training_list_input, training_list_output])

    training_list = training_list_3
    RNA_structure = [100, 20, len(training_list_3[0][1])]
    comprobation_list = training_list_3[55:]
    training_list = training_list_3[:55]
                        
        
    if Execution_mode == 1:
        convergence_magnitud=4

    else:
        file_name = "./RNA/Weights_datasets/Perceptron-3-25_04_2024-00_16_45.txt"


else:
    print("No training list selected")
    exit()
        


RNA = Perceptron(len(training_list[1][0]), RNA_structure, 'sigmoid',learning_rate=0.1, lambda_= 25*math.pow(10,-8))


for training_item in comprobation_list:
    print("Initial hipotesis for case Y = ", training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )

if Execution_mode == 1:
    J_evolution = RNA.Training(training_list, convergence_magnitud=convergence_magnitud, debug=Debug)
else:
    RNA.Set_weights(file_name)

print("\n ---- RNA calibrated ---- \n")

for training_item in comprobation_list:
    print("Final hipotesis for case Y = ", training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )

if Save_weights:
    file_name = "./RNA/Weights_datasets/Perceptron-"+ str(Exercise_id) +"-" + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".txt"
    RNA.Export_weights_to_file(file_name)
    print("Weights saved in ", file_name)





