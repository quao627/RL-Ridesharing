
import math
from gridmap import GridMap
from passenger import Passenger
from car import Car
from util import Util
from dqn import DQN
from networkx.algorithms.shortest_paths import shortest_path_length

class PairAlgorithm:


    def greedy_fcfs(self, grid_map):
        passengers = grid_map.passengers
        cars = grid_map.cars
        action = [0]*len(passengers)
        for i, p in enumerate(passengers):
            min_dist = math.inf
            assigned_car = None
            for j, c in enumerate(cars):
                dist = Util.cal_dist(p.pick_up_point, c.position)
                if dist < min_dist:
                    min_dist = dist
                    assigned_car = j
            action[i] = assigned_car

        return action
    

def ManhattanDistance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


class NewPairAlgorithm:

    def greedy_fcfs(self, grid_map):
        selected_cars = set()
        passengers = grid_map.passengers
        cars = grid_map.cars
        action = [-1]*len(passengers)
        for i, p in enumerate(passengers):
            min_dist = math.inf
            assigned_car = None
            for j, c in enumerate(cars):
                if j in selected_cars:
                    continue
                dist = shortest_path_length(grid_map.network, source=p.pick_up_point, target=c.position, weight='weight')
                if dist < min_dist:
                    min_dist = dist
                    assigned_car = j
            if assigned_car != None:
                selected_cars.add(assigned_car)
                action[i] = assigned_car

        return action
                    


if __name__ == '__main__':
    algorithm = PairAlgorithm()
    grid_map = GridMap(0, (5,5), 3, 3)
    grid_map.init_map_cost()
    grid_map.visualize()
    print(grid_map)
    algorithm.greedy_fcfs(grid_map)
    print(grid_map)
