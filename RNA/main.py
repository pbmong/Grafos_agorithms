# Version: 1.0
from Perceptron import Perceptron
import math

import os
from datetime import datetime

import imageio.v3 as iio
#Select the training list and the RNA structure

Execution_mode = 1 # 1 for training, 2 to charge weights
Exercise_id = 3
Save_weights = True
Debug = 1

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
    convergence_magnitud=4

elif Exercise_id == 2:
    if Execution_mode == 1:
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
        convergence_magnitud=4

    elif Execution_mode == 2:
        file_name = "./RNA/Weights_datasets/Perceptron-2-23_04_2024-00_14_16.txt"
        weights_file = open(file_name,"r")
        weights_str = weights_file.read()
        weights_file.close()

        #TODO: Parse weights

elif Exercise_id == 3:
    if Execution_mode == 1:
        images_folder = "./RNA/training_resouces/characters/"
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
            if image_path[0][0] == 'a':
                training_list_output = [1,0,0,0,0,0]
            elif image_path[0][0] == 'b':
                training_list_output = [0,1,0,0,0,0]
            elif image_path[0][0] == 'c':
                training_list_output = [0,0,1,0,0,0]
            elif image_path[0][0] == 'd':
                training_list_output = [0,0,0,1,0,0]
            elif image_path[0][0] == 'e':
                training_list_output = [0,0,0,0,1,0]
            elif image_path[0][0] == 'f':
                training_list_output = [0,0,0,0,0,1]
            
            training_list_3.append([training_list_input, training_list_output])

        training_list = training_list_3
        RNA_structure = [40, len(training_list_3[0][1])]
        comprobation_list = [training_list_3[1],
                            training_list_3[12],
                            training_list_3[23],
                            training_list_3[34],
                            training_list_3[45],
                            training_list_3[56]]
        convergence_magnitud=4


else:
    print("No training list selected")
    exit()



RNA = Perceptron(len(training_list[1][0]), RNA_structure, 'sigmoid',learning_rate=0.1, lambda_= 25*math.pow(10,-8))

for training_item in comprobation_list:
    print("Initial hipotesis for case", training_item[0], training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )

# To review biass point
J_evolution = RNA.Training(training_list, convergence_magnitud=convergence_magnitud, debug=Debug)

for training_item in comprobation_list:
    print("Hipotesis for case", training_item[0], training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )

if Save_weights:
    file_name = "./RNA/Weights_datasets/Perceptron-"+ str(Exercise_id) +"-" + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".txt"
    weights_file = open(file_name,"w") #np.load
    weights_file.write(str(RNA.Get_weights()))
    weights_file.close()





