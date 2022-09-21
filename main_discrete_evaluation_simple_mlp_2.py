from cProfile import run
from hashlib import new


from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import matplotlib.pyplot as plt
import time

import numpy as np
from itertools import combinations

import os
import math
import config as cf
import numpy as np
import json
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env

from robot_map2 import Map
from stable_baselines3.common.callbacks import CheckpointCallback

from a_star import AStarPlanner
from shapely.geometry import  LineString



class RobotEnv(Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.max_number_of_points = cf.model["map_size"] - 2 
        self.action_space = MultiDiscrete([2, cf.model['max_len'] - cf.model['min_len'], cf.model['max_pos'] - cf.model['min_pos']])  # 0 - increase temperature, 1 - decrease temperature
        self.observation_space = Box(low=0, high=self.max_number_of_points, shape=(self.max_number_of_points*3,), dtype=np.int8)
        #self.observation_space = Box(low=0, high=255, shape=(cf.model["map_size"], cf.model["map_size"], 1), dtype=np.uint8)

        self.sx = 1.0  # [m]
        self.sy = 1.0  # [m]
        self.gx = cf.model["map_size"] - 2  # [m]
        self.gy = cf.model["map_size"] - 2 # [m]
        #self.map_builder = Map(cf.model["map_size"])
        self.bonus = 0
        self.all_states = []
        self.all_fitness = []

        self.grid_size = 1  # [m]
        self.robot_radius = 0.1  # [m]
       
        self.episode = 0
        self.max_steps = 40   
        self.evaluation = False
        self.fitness = 0


    def generate_init_state(self):
        self.state = np.zeros((self.max_number_of_points, 3))
        random_position = 0

        ob_type = np.random.randint(0, 2)
        value = np.random.randint(cf.model["min_len"], cf.model["max_len"] + 1)
        position = np.random.randint(cf.model["min_pos"], cf.model["max_pos"] + 1)
        self.state[random_position]  = np.array([ob_type, value, position])
        self.old_location = position
        self.position_explored = [[ob_type, position]]
        self.sizes_explored = [value]

    def get_length(self, rx, ry):
        total_len = 0
        
        for i in range(1, len(rx)):
            len_ = math.sqrt((rx[i] - rx[i-1])**2 + (ry[i] - ry[i-1])**2)
            total_len += len_

        return total_len



    def eval_fitness(self, map_points):
        ox = [t[0] for t in map_points]
        oy = [t[1] for t in map_points] 


        a_star = AStarPlanner(ox, oy, self.grid_size, self.robot_radius)  # noqa: E501

        rx, ry, _ = a_star.planning(self.sx, self.sy, self.gx, self.gy)

        path = zip(rx, ry)
        
        scenario_size  = self.get_size()
        if len(rx) > 2:
        
            test_road = LineString([(t[0], t[1]) for t in path])
            self.fitness = test_road.length
        else:
            self.fitness = -10

            self.done = True


        return self.fitness

    def get_size(self):
        num = 0
        for i in (self.state):
            if i[1] != 0:
                num += 1
        return num
                
 
    def step(self, action):
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)   

        if self.steps >= self.max_steps - 3:
            self.done = True

        map_builder = Map(cf.model["map_size"])
        map_points = map_builder.get_points_from_states(self.state)

        points_list = map_builder.get_points_cords(map_points)

        new_reward = self.eval_fitness(points_list) # - discount
        current_state = self.state.copy()

        improvement = new_reward - self.old_reward
        position = [action[0], action[2] + cf.model['min_pos']]
        value = action[1] + cf.model['min_len']

        if new_reward < 0:
            reward = -10
        else:

            if new_reward <= 60:
                reward = 0
            elif new_reward > 60 and new_reward <= 70:
                reward = 5
            elif new_reward > 70 and new_reward <= 80:
                reward = 50
            elif new_reward > 80:
                reward = 100
            elif new_reward > 90:
                reward = 200
            elif new_reward > 100:
                reward = 500

            if improvement > 0:
                reward += improvement*10

            if not(value in self.sizes_explored):
                reward += 5
                self.sizes_explored.append(value)

            if not(position in self.position_explored):
                reward += 10
                self.position_explored.append(position)
        

        self.old_reward = new_reward
        
        self.all_fitness.append(self.fitness)
        self.all_states.append(current_state)

        self.steps += 1

        info = {}

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        

        return np.array(obs, dtype=np.int8), reward, self.done, info

    def reset(self):

        #print(self.fitness)

        self.generate_init_state()

        map_builder = Map(cf.model["map_size"])
        map_points = map_builder.get_points_from_states(self.state)
        points_list = map_builder.get_points_cords(map_points)

        self.scenario_size = self.get_size() 

        default_reward = self.eval_fitness(points_list)

        
        self.old_reward = default_reward#bonus



        self.all_states = []
        self.all_fitness = []

        self.fitness = 0

        self.steps = 1

        self.done = False

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8)

    def render(self, scenario):

        #if self.done:

        fig, ax = plt.subplots(figsize=(12, 12))

        map_builder = Map(cf.model["map_size"])
        map_points = map_builder.get_points_from_states(scenario)
        points_list = map_builder.get_points_cords(map_points)



        road_x = []
        road_y = []
        for p in points_list:
            road_x.append(p[0])
            road_y.append(p[1])


        a_star = AStarPlanner(road_x, road_y, self.grid_size, self.robot_radius)  # noqa: E501

        rx, ry, _ = a_star.planning(self.sx, self.sy, self.gx, self.gy)

        path = zip(rx, ry)


        test_road = LineString([(t[0], t[1]) for t in path])
        fit = test_road.length

  

        ax.plot(rx, ry, '-r', label="Robot path")

        ax.scatter(road_x, road_y, s=150, marker='s', color='k', label="Walls")

        top = cf.model["map_size"]
        bottom = 0

        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.set_ylim(bottom, top)
        
        ax.set_xlim(bottom, top)
        ax.legend(fontsize=22)

        top = cf.model["map_size"] + 1
        bottom = - 1


        if os.path.exists(cf.files["img_path"]) == False:
                os.mkdir(cf.files["img_path"])

        fig.savefig(cf.files["img_path"] + str(self.episode) + "_" + str(fit) + ".png")

        fig.savefig("test.png")

        plt.close(fig)



    def set_state(self, action):
        #if action[0] == 0:
        obs_size = action[1] + cf.model['min_len']
        position = action[2] + cf.model['min_pos']

        return [action[0], obs_size, position]


        
def compare_states(state1, state2):
    similarity = 0
    if state1[0] == state2[0]:
        similarity += 1
        if abs(state1[1] - state2[1]) <= 5:
            similarity += 1
        if abs(state1[2] - state2[2]) <= 5:
            similarity += 1

    return similarity

def calc_novelty(old, new):
    similarity = 0

    total_states = (len(old))*3

    if len(old) > len(new):
        for i in range(len(new)):
            similarity += compare_states(old[i], new[i])
    elif len(old) <= len(new):
        for i in range(len(old)):
            similarity += compare_states(old[i], new[i])
    novelty = 1 - (similarity/total_states)
    return -novelty



if __name__ == "__main__":
    print("Starting...")

    final_results = {}
    final_novelty ={}
    scenario_list = []
    novelty_list = []

    m = 0 
    for m in range(3):

        environ = RobotEnv()

        #check_env(environ)


        # Save a checkpoint every 1000 steps
        checkpoint_callback_ppo = CheckpointCallback(save_freq=10000, save_path=cf.files["model_path"],
                                                name_prefix='rl_model_09-21_mlp'+str(m))
        log_path = cf.files["logs_path"]

        
        start = time.time()
        #'MlpPolicy'
    
        
        model = PPO('MlpPolicy', environ,verbose=True, batch_size=32, learning_rate=0.003, tensorboard_log=log_path)#, policy_kwargs=policy_kwargs)
        #learning_rate=0.003, batch_size=32, n_epochs=30, n_steps=2048,nt_coef=0.005, gamma=0.97,


        # Start training the agent
        model.learn(total_timesteps=200000, tb_log_name="ppo_mlp", callback=checkpoint_callback_ppo)  #, tb_log_name="a2c"
        print("Training time: {}".format(time.time() - start))
        # test the environment
    
        
        '''
        model_save_path = cf.files["model_path"] + "rl_model_09-19_mlp0_40000_steps.zip"
        model = PPO.load(model_save_path)
        '''

        
        episodes = 30
        environ.evaluation = True
        
        i = 0
        results = []
        while environ.episode < episodes:
            obs = environ.reset()
            done = False
            
            while not done:
                #obs = np.array([obs])
                action, _ = model.predict(obs)
                #action = action[0]
                #print(action)
                obs, rewards, done, info = environ.step(action)
            i += 1
            max_fitness = max(environ.all_fitness)
            max_index = np.argmax(environ.all_fitness)
            print(max_fitness)

            if (max_fitness > 60) or i >15:
                print(i)
                print("Round: {}".format(environ.episode))
                print("Max fitness: {}".format(max_fitness))
                
                #scenario = environ.state[:environ.t]
                scenario = environ.all_states[max_index]
                environ.render(scenario)
                scenario_list.append(scenario)
                environ.episode += 1
                results.append(max_fitness)
                i  = 0

        #results = environ.results
        

        final_results[str(m)] = results

        
        novelty_list = []
        for i in combinations(range(0, 30), 2):
            current1 = scenario_list[i[0]]
            current2 = scenario_list[i[1]]
            nov = calc_novelty(current1, current2)
            novelty_list.append(nov)
        novelty = abs(sum(novelty_list)/len(novelty_list))

        final_novelty[str(m)] = novelty

        scenario_list = []

        
        with open('2022-09-21-results-ppo_mlp.txt', 'w') as f:
            json.dump(final_results, f, indent=4)

        with open('2022-09-21-novelty-ppo_mlp.txt', 'w') as f:
            json.dump(final_novelty, f, indent=4)
        
        
