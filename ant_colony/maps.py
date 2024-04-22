class Map:
    width = 0
    length = 0
    cells = [[],[]]

    def __init__(self, width, length, cells, foods):
        self.width = width
        self.length = length
        self.cells = [[Map_cell(0, 0, False, None, 0, 1) for y in range(length)] for x in range(width)]
        
        for x in range(width-1):
            for y in range(length-1):
                wall = (cells[x][y][0] == 0) & (cells[x][y][1] == 0) & (cells[x][y][2] == 0)
                self.cells[x][y] = Map_cell(x, y, wall, None, 0, 1)

        for food in foods:
            self.cells[food[0]][food[1]].set_food(food[2])

    def get_cell(self, x, y):
        return self.cells[y][x]
    
class Map_cell:

    x = 0
    y = 0
    is_wall = False
    food = 0
    ant = False
    feromone = 0
    cost = 1

    def __init__(self, x, y, is_wall, ant, feromone, cost):
        self.x = x
        self.y = y
        self.is_wall = is_wall
        self.feromone = feromone
        self.cost = cost

    def set_food(self, food):
        self.food = food