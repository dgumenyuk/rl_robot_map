 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import time
import numpy as np
class Map:
    """Class that conducts transformations to vectors automatically,
    using the commads "go straight", "turn left", "turn right".
    As a result it produces a set of points corresponding to a road
    """

    def __init__(self, map_size):
        self.map_size = map_size
        self.max_x = map_size
        self.max_y = map_size
        self.min_x = 0
        self.min_y = 0

        self.init_pos = [0, 1]

        self.map_points = []

        # self.current_pos = [self.init_pos, self.init_end]
        self.all_map_points = np.ones((self.map_size, self.map_size))
        self.current_level = 1

        self.create_init_box()

    def create_init_box(self):
        """select a random initial position from the middle of
        one of the boundaries
        """

        self.all_map_points[0][:]  = 0
        for i in range(1, self.map_size):
            self.all_map_points[i][0] = 0
            self.all_map_points[i][self.map_size - 1] = 0

        self.all_map_points[-1][:] = 0

        
        self.all_map_points[-2][:3] = 0
        self.all_map_points[-3][:3] = 0

        self.all_map_points[1][-3:] = 0
        self.all_map_points[2][-3:] = 0
        


        return

    def horizontal(self, distance, position):

        new_points = []

        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[-init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_left = [init_pos[0] - i, init_pos[1]]
            point_right = [init_pos[0] + i, init_pos[1]]
            if self.point_valid(point_left):
                self.all_map_points[-point_left[1]][point_left[0]] = 0
            if self.point_valid(point_right):
                self.all_map_points[-point_right[1]][point_right[0]] = 0
                #new_points.append(point_right)

        self.current_level += 1

        return new_points

    def vertical(self, distance, position):

        new_points = []

        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[-init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_down = [init_pos[0], init_pos[1] - i]
            point_up = [init_pos[0], init_pos[1] + i]
            if self.point_valid(point_down):
                self.all_map_points[-point_down[1]][point_down[0]] = 0
            if self.point_valid(point_up):
                self.all_map_points[-point_up[1]][point_up[0]] = 0

        self.current_level += 1

        return new_points, self.discount

    def point_valid(self, point):
        if (self.in_polygon(point)) or self.point_out_of_bounds(point):
            self.discount += 1
            return False
        else:
            return True


    def point_out_of_bounds(self, a): 
        if (0 <= a[0] and a[0] < self.max_x) and (0 <= a[1] and a[1] < self.max_y):
            return False
        else:
            #print("OUT OF BOUNDS {}".format(a))
            return True


    

    '''

    def point_out_of_bounds(self, a):
        if (0 < a[0] and a[0] < self.max_x ) and (0 < a[1] and a[1] < self.max_y - 4):
            return False
        else:
            return True
    '''


    def in_polygon(self, a):
        """checks whether a point lies within a polygon
        between current and previous vector"""
        thresh = 3
        if (a[0] < thresh ) and (a[1] < thresh) :
            #print("IN POLYGON1 {}".format(a))
            return True
        elif ((a[0] > self.max_x - thresh ) and (a[1] >  self.max_y - thresh)):
            #print("IN POLYGON2  {}".format(a))
            return True
        else:
            return False

    def get_points_cords(self, points):
        """returns a list of points that are in the polygon"""
        cords = []
        for i, row in enumerate(reversed(points)):
            for j, point in enumerate(row):
                if point == 0:
                    cords.append([j, i])
        
        to_remove = [[1, 1], [1, 2], [2, 1], [2, 2], [self.max_x - 2, self.max_y - 2], [self.max_x - 3, self.max_y - 3], [self.max_x - 3, self.max_y - 2], [self.max_x - 2, self.max_y - 3]]
        for r in to_remove:
            cords.remove(r)
        

        return cords


    def get_points_from_states(self, states):



        self.current_level = 2


        self.create_init_box()

        self.discount = 0

        tc = states
        for state in tc:
            #self.build_tc(self.all_map_points)
            action = int(state[0])
            if action == 0:
                self.horizontal(int(state[1]), int(state[2]))
            elif action == 1:
                self.vertical(int(state[1]), int(state[2]))
            else:
                print("ERROR")

        #print("OBTAINED POINTS", points)
        return self.all_map_points

    '''

    def change_state(self, obs_type, size, x, y):

        self.current_level = y
        
        if obs_type == 0:
            self.horizontal(size, x)
        elif  obs_type == 1:
            self.vertical(size, x)
        else:
    '''







    def build_tc(self, points):


        #time_ = str(int(time.time()))

        fig, ax = plt.subplots(figsize=(12, 12))
        # , nodes[closest_index][0], nodes[closest_index][1], 'go'
        road_x = []
        road_y = []
        for p in points:
            road_x.append(p[0])
            road_y.append(p[1])

        ax.plot(road_x, road_y, 'yo--', label="Road")

        top = self.map_size
        bottom = 0

        ax.set_title("Test case fitenss ", fontsize=17)

        ax.set_ylim(bottom, top)
        
        ax.set_xlim(bottom, top)

        #fig.savefig(".\\Test\\"+ time_+ "+test.jpg")
        fig.savefig("test_inside.png")

        ax.legend()
        plt.close(fig)




def read_schedule(tc, size):
    time_ = str(int(time.time()))
    fig, ax1 = plt.subplots(figsize=(12, 12))
    car_map = Map(size)
    for state in tc:
        action = tc[state]["state"]
        print("location:", car_map.current_pos)
        if action == "straight":
            car_map.go_straight(tc[state]["value"])
            print(tc[state]["value"])
            x, y = car_map.position_to_line(car_map.current_pos)
            ax1.plot(x, y, "o--y")
        elif action == "left":
            car_map.turn_left(tc[state]["value"])
            print(tc[state]["value"])
            x, y = car_map.position_to_line(car_map.current_pos)
            ax1.plot(x, y, "o--y")
        elif action == "right":
            car_map.turn_right(tc[state]["value"])
            print(tc[state]["value"])
            x, y = car_map.position_to_line(car_map.current_pos)
            ax1.plot(x, y, "o--y")
        else:
            print("Wrong value")

    top = size
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xlim(bottom, top)
    #plt.yticks(np.arange(bottom, top + 1, 1.0), fontsize=12)
    #plt.grid(b=True, which="major", axis="both")

    #ax1.legend(fontsize=14)
    fig.savefig(".\\Test\\"+ time_+ "+test2.jpg")
    plt.close(fig)






if __name__ == "__main__":


    my_map = Map(200)
    points = my_map.all_map_points


    ox = [t[0] for t in points]
    oy = [t[1] for t in points]

    plt.plot(ox, oy, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

