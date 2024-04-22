import random
import math

max_feromone = 10

class Ant_colony:
    ants = list()
    ants_number = 0
    pos_x = 0
    pos_y = 0

    def __init__(self, ants_number, x, y):
        # Create a list of ants
        for i in range(ants_number):
            self.ants.append(Ant(i+1, x, y, [x, y]))
        self.ants_number = ants_number
        self.pos_x = x
        self.pos_y = y

class Ant:
    id = 0 # Ant identifier
    location = [-1, -1] # Ant current location
    food = False # Ant has food
    food_location = None # Food location
    colony_location = [-1, -1]
    direction = None

    def __init__(self, id, x, y, colony_location,direction = None, food = False):
        self.id = id
        self.location= [x, y]
        self.food = food
        self.colony_location = colony_location
        self.direction = direction
        
    def move(self, environment, direction = None):
        
        food_found_flag = False

        if self.direction == self.location:
            self.direction = None

        try:
            # Check if there are feromones in the environment
            if self.food == False:
                better_feromone = 0
                for x in range(3):
                    for y in range(3):
                        if environment[x][y].feromone > 0:
                            if environment[x][y].feromone > better_feromone:
                                better_feromone = environment[x][y].feromone
                                direction = [environment[x][y].x, environment[x][y].y]
                            break
        except Exception as e:
             self.direction = None

        if (direction != None) and (self.food == False) and (self.food_location == None):
            self.direction = direction

        # If there is a direction to follow
        if self.direction != None:
            try:
                mod = math.pow((self.direction[0] - self.location[0]),2) + math.pow((self.direction[1] - self.location[1]),2)
                direction_module = math.sqrt(float(mod))

                inc_x = round((self.direction[0] - self.location[0]) / direction_module)
                inc_y = round((self.direction[1] - self.location[1]) / direction_module)

                if environment[inc_x + 1][inc_y + 1].is_wall == False:
                    
                    self.location[0] = self.location[0] + inc_x
                    self.location[1] = self.location[1] + inc_y

                    if (environment[inc_x][inc_y].food > 0) and (self.food == False):
                        food_found_flag = True

                    if (self.location[0] == self.direction[0]) and (self.location[1] == self.direction[1]):
                        self.direction = None
                else:
                    self.direction = None

            except Exception as e:
                self.direction = None

        else: # Random movement
            while True:
                random_direction = [random.randint(-1, 1), random.randint(-1, 1)]
                destiny_cell = environment[random_direction[0] + 1][random_direction[1] + 1]
                if random_direction != [0, 0] and destiny_cell.is_wall == False:
                    break

            self.location[0] += random_direction[0]
            self.location[1] += random_direction[1]

            if (environment[random_direction[0] + 1] [random_direction[1] + 1].food > 0) and (self.food == False):
                food_found_flag = True
        
        # Check if food have been found
        if food_found_flag:
            self.food = True
            self.food_location = [self.location[0], self.location[1]]
            self.direction = [self.colony_location[0], self.colony_location[1]]

        # Check if the ant is in the colony
        elif (self.location[0] == self.colony_location[0]) and (self.location[1] == self.colony_location[1]) and (self.food == True):
            self.food = False
            self.direction = [self.food_location[0], self.food_location[1]]

        # Generate feromone
        if self.food_location != None:
            return True
                

            