#Main Imports
import gym
import numpy as np
import random
import numpy as np
import time
import utils.camera as cam
import matplotlib.pyplot as plt
import cv2
import math

#utils import
from robot import Robot
from itertools import count
from collections import namedtuple, deque

#torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Named tuple declaration to store the transitions in Deque memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#If GPU is available use the following for processing the transition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set the hyperparameters for the network training procedure
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


class RobotEnv(gym.Env):
    
    def __init__(self)->None:
        
        self.robot = Robot()
        self.camera = cam.Camera()
        self.grid_size = (3, 3)
        self.start_pos = (0, 0)
        self.end_pos = (2, 2)
        self.trap_pos = (1, 1)
        self.actions = [0, 1, 2, 3]  # [Right,Left, Backward, Forward]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.action_keywords = {0: 'Right', 1: 'Left', 2: 'Backward', 3: 'Forward'}

        
    def reset(self)->tuple:
        '''go back to 50,50 position in Robot coordinate'''
        
        while self.robot.is_cam_ready() == '0':
            time.sleep(0.1)       
                  
        self.robot.reset() #move to defined home coordinates in (x,y) as per rapid  
        self.agent_pos = self.start_pos #set the agent to (0,0) position
        self.done = False
        
        return self.agent_pos
    
    def take_image_assign(self)->tuple:
        '''Get the image when status is do_check status is 1 and based on pixel (x,y) of coin 
           get the cell position to use for the state space say (0,0) for pixel location in 
           top left of the image plane.'''
           
        if self.get_cam_status() == '1':
            image_roi = self.get_image()
            x,y = self.find_obj_centre(image = image_roi)
            cell_x,cell_y = self.find_cell_index(x,y)
            print("\n Cell location got from pixel (x,y) is (x: {}, y: {})".format(cell_x,cell_y))
            return cell_x,cell_y
          
    def check_valid_action(self,action:int,cell_x:int,cell_y:int)->bool:
        '''Check if the action belonging to a cell is valid or not ..
            If it is valid return True else return False '''
        
        self.print_action(action)
        
        if action == 0: #right action
            if (cell_x == 0):
                return False
            else:
                return True
        if action == 1: #Left action
            if (cell_x == 2):
                return False
            else:
                return True
        if action == 2: #Backward action
            if (cell_y == 0):
                return False
            else:
                return True
        if action == 3: #Forward action
            if (cell_y == 2):
                return False
            else: 
                return True
                
    def step(self,action:int)->tuple:
        
        if self.done:
            raise ValueError("Episode has already terminated. Please reset the environment.")
        
        x,y = self.agent_pos #get x,y of agent position
    
        if action == 0:  # Move right
            
            cell_x,cell_y = self.take_image_assign() #get corresponding cell position
            action_check = self.check_valid_action(action,cell_x,cell_y)
            self.move(action=action,action_check=action_check) #move robot 
            cell_x_new,cell_y_new = self.take_image_assign() #get corresponding cell position again after movement based on action
            
            if action_check == True:
                #x,y = cell_x-1, cell_y
                x,y = cell_x_new, cell_y_new
            else:
                #x,y = cell_x, cell_y
                x,y = cell_x_new, cell_y_new
          
        elif action == 1:  # Move left
            
            cell_x,cell_y = self.take_image_assign() #get corresponding cell position
            action_check = self.check_valid_action(action,cell_x,cell_y)
            self.move(action=action,action_check=action_check) #move robot 
            cell_x_new,cell_y_new = self.take_image_assign() #get corresponding cell position again after movement based on action
            
            if action_check == True:
                #x,y = cell_x+1 , cell_y
                x,y = cell_x_new, cell_y_new
            else:
                x,y = cell_x_new, cell_y_new
            
        elif action == 2:  # Move backward'
            
            cell_x,cell_y = self.take_image_assign() #get corresponding cell position
            action_check = self.check_valid_action(action,cell_x,cell_y)
            self.move(action=action,action_check=action_check) #move robot 
            cell_x_new,cell_y_new = self.take_image_assign() #get corresponding cell position again after movement based on action
            
            if action_check == True:
                #x,y = cell_x, cell_y-1
                x,y = cell_x_new, cell_y_new
            else:
                x,y = cell_x_new, cell_y_new
            
        elif action == 3:  # Move forward
            
            cell_x,cell_y = self.take_image_assign() #get corresponding cell position
            action_check = self.check_valid_action(action,cell_x,cell_y)
            self.move(action=action,action_check=action_check) #move robot 
            cell_x_new,cell_y_new = self.take_image_assign() #get corresponding cell position again after movement based on action
            
            if action_check == True:
                x,y = cell_x_new, cell_y_new
                #x,y = cell_x, cell_y+1
            else:
                x,y = cell_x_new, cell_y_new

        #update the moved state to the agent 
        self.agent_pos = (x, y)

        #can tweak the reward and check if having high reward for goal makes any difference or not
        #check if new agent pos is terminal or not and assign rewards
        
        if self.agent_pos == self.end_pos:
            reward = 10
            self.done = True
            
        elif self.agent_pos == self.trap_pos:
            reward = -5
            self.done = True
            
        else:
            reward = -1

        return self.agent_pos, reward, self.done, {}
    

    def find_cell_index(self,x:int,y:int,image_width:int=200,image_height:int=200)->tuple:
        '''find the cell index based on camimage_width and image_height and 
        (x,y) pixel coordinate return list of cells example : (0,1)'''
            
        grid_width = image_width // 3
        grid_height = image_height // 3
        #print("Pixel x coordinate", x)
        #print("Pixel y coordinate", y)
        cell_x = int(x) // grid_width
        cell_y = int(y) // grid_height
            
        return cell_x, cell_y
    
    def move(self,action:int,action_check:bool)->None:
        '''Moves the Robot by sending appropriate commands to RAPID routine'''
        while self.robot.is_cam_ready() == False:
            time.sleep(0.1)
        self.robot.move_rob(action,action_check)
        
    def get_image(self):
        '''Get the roi image after cropping and plot it .'''
        image_roi =  self.camera.get_rgb()
        return image_roi
    
    def get_cam_status(self)->str:
        '''Get the boolean string ('1' or '0') to check if the camera is ready'''
        status = self.robot.is_cam_ready()
        while status == '0': #keep on listening for the socket signal when receiving
            status = self.robot.is_cam_ready()
            time.sleep(0.1)
            #print("The status is : {}".format(status))
        return status
        
    def find_obj_centre(self,image,threshold=100)->tuple:
            
        '''Function to find pixel x,y from coin using gray image and cv2 
        image processing optinally can retrieve radius as well'''
            
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        ret,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        
        #modify to show grid and pixel centre of coin as title
        plt.imshow(thresh,cmap='gray')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close() 
        
        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse=True)[:1]
        cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
            
        return int(x), int(y)
    
    def print_action(self, action:int)->str:
        '''Just a print function to display the corresponding action given a action integer for better
            traceability'''
        
        action_keyword = self.action_keywords[action]
        print("\n Action sampled is : {}".format(action_keyword))
        
    def generate_random_action(self):
        '''
        Temporary solution , later implement selection_action function based on eps value below.
        generates a random number from 0-3 to take random action.
        For unbiased sampling make thresh 0.5 and to favor the odds of making left or forward
        movement more make the thresh bigger than 0.5 (default 0.6).
        Higher the value more is the chance of sampling left or forward action.
        '''
        
        random_number = random.random()
        
        if random_number < 0.6:
            return random.choice([1,3])
        else:
            return random.choice([0,2])

    def close(self):
        pass

    def action_space(self):
        return len(self.actions)

    def observation_space(self):
        return self.grid_size
        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)       
    
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def optimize_model():
    
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()  
    
#needs to be modified after testing the random action and memory allocation for deque when optmizing using policy net parameters.
def select_action(state):
    
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #maybe update to always favor forward and left more than backward and right.
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

#To keep track of number of transitions required to make the goal. 
steps_done = 0 
episode_durations = []

#Instantiate the model for policy and target
policy_net = DQN(2, 4).to(device)
target_net = DQN(2, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())   

#Declare optimizer and replay memory
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

if __name__ == "__main__":
    
    num_episodes = 3
    env = RobotEnv()
    
    for i_episode in range(num_episodes):
        #Init the env and get the state 
        state = env.reset() # so my state now is (0,0)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            
            action = env.generate_random_action() #gonna be an integer in (0,1,2,3)
            observation, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device) #put reward also in CUDA device to make all the tensors in GPU.
            
            if done:
                next_state = None
            else:
                next_state =  torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            #assign current state to next state
            state = next_state
            
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            
            if done:
                episode_durations.append(t + 1)
                print("\n The episode is done going back to reset (0,0) !!!")
                print("\n The length of the episode is : {}".format(episode_durations[i_episode]))
                break
            
    print("\n The final number of transitions stored in the memory deque is : {}".format(len(memory)))
            
            
            
        
    

    
        
        
