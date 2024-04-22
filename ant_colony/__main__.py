import os,sys
import time
import random

from ant_colony import Ant_colony
from maps import Map
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt


#Define colony parameters
ants_number = 50
colony_pos_x = 1
colony_pos_y = 1

#Define food parameters
foods = [[50,50,10],[50,51,10],[50,52,10],[50,53,10],
         [51,50,10],[51,51,10],[51,52,10],[51,53,10],
         [52,50,10],[52,51,10],[52,52,10],[52,53,10]]

#Define simulation parameters
interations = 200
random_direction_iteration = 10

# import map image
map_path = "maps/Simple_obstacles.png"

# read an image 
im = iio.imread(map_path)
map_array = np.array(im)
print(map_array.shape)

# Create map
width = map_array.shape[0]
length = map_array.shape[1]
map = Map(width, length, map_array, foods)

# Create map image
map_image = np.zeros((width,length,3), dtype=np.uint8)

plt.ion()
fig1, ax1 = plt.subplots()
data_image = ax1.imshow(map_image)

# Create a new instance of the AntColony class
ant_colony = Ant_colony(ants_number, colony_pos_x, colony_pos_y)

for i in range(interations):

    # Clean feromones in map
    for x in range(width):
        for y in range(length):
            map.cells[x][y].feromone -= 1

    direction = None
    # Move the ants
    for ant in ant_colony.ants:
        if (i % random_direction_iteration == 0):
            direction = [random.randint(0, width), random.randint(0, length)]
        try:
            ant_environment = map.cells[ ant.location[0]-1 : ant.location[0]+2]
            ant_environment = [ant_environment[0][ ant.location[1]-1 : ant.location[1]+2] ,
                               ant_environment[1][ ant.location[1]-1 : ant.location[1]+2] ,
                               ant_environment[2][ ant.location[1]-1 : ant.location[1]+2] ]
            food_found = ant.move(ant_environment, direction)
            
        except IndexError as e:
            print(e)
            food_found = ant.move(ant_environment, [colony_pos_x,colony_pos_y])
            continue

        if food_found:
            map.cells[ant.location[0]][ant.location[1]].feromone = 10
            map.cells[ant.location[0]][ant.location[1]].food -= 1

    # Update the map image
    for x in range(width):
        for y in range(length):
            if map.cells[x][y].is_wall:         # if the cell is a wall
                map_image[x][y] = [0,0,0]

            elif map.cells[x][y].food > 0:      # if the cell has food
                map_image[x][y] = [25*map.cells[x][y].food, 25*map.cells[x][y].food ,0]

            elif map.cells[x][y].feromone > 0:  # if the cell has feromone
                map_image[x][y] = [0, 25*map.cells[x][y].feromone,0] 

            else:                               # if the cell is empty
                map_image[x][y] = [255,255,255]

    # Update ants in the map image
    for ant in ant_colony.ants:
        map_image[ant.location[0]][ant.location[1]] = [255,0,0]

    map_image[colony_pos_x][colony_pos_y] = [0,255,0]

    data_image.set_data(map_image)
    fig1.canvas.flush_events()

    #time.sleep(0.01)


input("Press Enter to finish")

       
