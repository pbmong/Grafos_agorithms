# Version: 1.0
from Perceptron import Perceptron
import math

training_list_1=[
    [[0,0],[0,0]],
    [[0,1],[0,1]],
    [[1,0],[0,1]],
    [[1,1],[1,1]]
]
RNA_1 = [3, 2]

training_list_2=[
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]]
]
RNA_2 = [2, 1]

training_list = training_list_1
RNA_structure = RNA_1

RNA = Perceptron(len(training_list[1][0]), RNA_structure, 'sigmoid',learning_rate=0.01, lambda_= 25*math.pow(10,-8))

for training_item in training_list:
    print("Initial hipotesis for case", training_item[0], training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )

# To review biass point
RNA.Training(training_list, iterations=50000, debug=False)

for training_item in training_list:
    print("Hipotesis for case", training_item[0], training_item[1] ,"] is ", RNA.Calculate_output(training_item[0]) )




