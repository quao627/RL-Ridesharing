#from algorithm import PairAlgorithm
from networkx.algorithms.shortest_paths import shortest_path_length

class NewEnvironment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        #self.algorithm = PairAlgorithm()
                 
                 
    def reset(self):
        
        self.grid_map.cars = []
        self.grid_map.passengers = []
        self.grid_map.add_passenger(self.grid_map.num_passengers)
        self.grid_map.add_cars(self.grid_map.num_cars) 
          
                        
    def step(self, action, mode):
        
        grid_map = self.grid_map
        cars = grid_map.cars
        passengers = grid_map.passengers
        reward = [0]*len(passengers)
        done = False
        
        
        reward = []

        for pax in passengers:
            pax.trip_time = shortest_path_length(grid_map.network, source=pax.pick_up_point, target=pax.drop_off_point, weight='weight')

        total_trip_time = sum([pax.trip_time for pax in passengers])

        for i, act in enumerate(action[0]):
            passenger = passengers[i]
            if act == -1:
                reward.append(0)
                continue
            car = cars[act]
            passenger.wait_time = shortest_path_length(grid_map.network, source=car.position, target=passenger.pick_up_point, weight='weight')
            reward.append((passenger.trip_time - passenger.wait_time)/total_trip_time)
  
                    
        return reward
    
    
            

                    


