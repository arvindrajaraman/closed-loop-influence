import copy
from gym import Env
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn

from device import device
from env_setup import *
from lqr import get_LQR_control, get_LQR_cost, get_LQR_control_sample
from build_human_action_set import build_human_action_set

# @title Simulated human internal model dynamics
# @title Simulated human internal model dynamics
def update_human_internal_dynamics_model(traj_snippet,\
                              human_obs_snippet,\
                              human_action_snippet,\
                              A_int_np, B_int_np,\
                              human_mode, eta):
    """Update the simulated human's internal robot model using gradient decent"""

    # compy the A and B tensor
    A_int_tensor = nn.parameter.Parameter(torch.tensor(A_int_np, device = device))
    B_int_tensor = nn.parameter.Parameter(torch.tensor(B_int_np, device = device))

    # human's internal loss = state prediction error
    human_int_loss = 0
    for i in range(1):
        x_0 = traj_snippet[i]
        u_0 = human_action_snippet[i]
        # human's desired state
        x_1_int = torch.matmul(A_int_tensor, torch.tensor(x_0, device = device)) + \
        torch.matmul(B_int_tensor, torch.tensor(u_0, device = device))
        # print(x_1_int)
        # print(human_obs_snippet[0])
        human_int_loss += torch.norm(x_1_int - torch.tensor(human_obs_snippet[i],device = device)) **2
    human_int_loss.backward()
    #print(human_int_loss)
    # we mask un-necessary gradient
    A_int_grad = A_int_tensor.grad.cpu().numpy() * A_grad_mask
    B_int_grad = B_int_tensor.grad.cpu().numpy() * B_grad_mask

    #print(u_drift_int_grad)
    d_A_int       = eta * A_int_grad
    d_B_int       = 4 * eta * B_int_grad

    #print(d_B_int)
    # update the internal model
    if human_mode == 'gradient_decent':
        A_int_np_new = A_int_tensor.detach().cpu().numpy() -    d_A_int
        B_int_np_new = B_int_tensor.detach().cpu().numpy() -    d_B_int

    if human_mode == 'gradient_decent_threshold':
        if abs(d_B_int[1,0]) < 0.002:
            d_B_int[1,0] = 0.0

        A_int_np_new = A_int_tensor.detach().cpu().numpy() -    d_A_int
        B_int_np_new = B_int_tensor.detach().cpu().numpy() -    d_B_int


    nX = A_int_tensor.shape[0]
    nU = B_int_tensor.shape[1]  
    B_int_np_new[1,0] = B_int_np_new[1,0]
    # clamp the updated A and B matrix
    for i in range(nX):
        for j in range(nU):
            B_int_np_new[i,j] = max(min(B_int_np_new[i,j], 1), 0.0)  
    for i in range(nX):
        for j in range(nU):
            A_int_np_new[i,j] = max(min(A_int_np_new[i,j], 1), 0.0)


    return A_int_np_new, B_int_np_new

#@title Human class
class HumanRobotEnv(Env):
    def __init__(self, robot_mode, alpha, human_type, is_updating_internal_model, human_lr) :
        # the mode of the robot (active or passive)
        self.robot_mode = robot_mode
        # the blending policy of the robot
        self.alpha = alpha
        # (modeled human, nn human)
        self.human_type = human_type
        # if human can updates the internal model in modeled human
        self.is_updating_internal_model = is_updating_internal_model
        self.human_lr = human_lr
        # ground truth environment dynamics (linear dynamics with LQR control)
        self.A_t = None
        self.B_t = None
        self.Q_t = None
        self.R_t = None
        self.u_drift_t = None
        # robot goal set
        self.ref_path = None
        self.nX = None
        self.nU = None
        self.episode_length = None
        # actions that the robot can take
        self.action_space = None
        self.human_action_set = None
        self.robot_action_set = None
        # observation recieved by the robot
        self.observation_space = Box(low=np.array([-2, -2, 0.0]), \
                                     high=np.array([2,  2, 1.0]), dtype=float)
        self.human_internal_model_dynamics_NN = None
        # Initialize the human state
        self.physical_state = None
        self.mental_state = None
        self.state = None
        
        # If true, the input action to step is the real human action
        self.real_human_mode   = False
        self.robot_action_mode = 'addon'
        self.noisy_human = False
        self.stochastic_human = False
        # goal index
        self.goal_index = 1
        self.step_count = 1
        # Initialize the traj, state list to save the data
        self.current_demo_state_traj              = []
        self.current_demo_human_action_traj       = []
        self.current_demo_human_obs_traj          = []
        self.current_demo_robot_action_traj       = []
        self.current_demo_human_mental_state_traj = []
        self.current_demo_reward_traj             = []
        self.current_demo_task_reward_traj        = []
        self.current_demo_action_reward_traj      = []
        self.xg_demo_hist = []
        self.xg_demo_index_hist = []
        self.current_demo_opt_action_traj = []
        self.current_demo_human_action_opt_traj = []
    
    def set_environment(self, A_t, B_t, Q_t, R_t, u_drift_t, ref_path, sequence_length):
        """
        Set the environment dynamics and the goal set
        """
        self.A_t = A_t
        self.B_t = B_t
        self.Q_t = Q_t
        self.R_t = R_t
        self.u_drift_t = u_drift_t
        self.ref_path = ref_path
        self.nX = A_t.shape[0]
        self.nU = B_t.shape[1]
        self.episode_length = sequence_length
    
    def set_action_set(self, human_action_set, robot_action_set):
        """
        Set the action set of the human and robot
        """
        self.action_space = Discrete(len(robot_action_set))
        self.human_action_set = human_action_set
        self.robot_action_set = robot_action_set
    
    def set_human_internal_model(self, dynamics_model):
        """
        Set the learned human model
        """
        self.human_internal_model = dynamics_model
    
    def set_human_state(self, physical_state, mental_state):
        """
        Set the human state
        """
        self.physical_state = physical_state
        self.mental_state = mental_state
        self.state = \
        np.concatenate((self.physical_state, self.mental_state), 0).reshape(physical_state.shape[0] + mental_state.shape[0])
    
    def update_goal(self):
        """
        Track the human's goal
        """
        #print('Current model physical_state is:', self.physical_state)
        goal_reach_threshold = 0.05
        d_2_curr_goal = np.linalg.norm(self.ref_path[self.goal_index,:].reshape(3,1) - self.physical_state)
        #print('Current  goal state is:', self.ref_path[self.goal_index,:].reshape(3,1))
       # print('Current goal index: ', self.goal_index)
        if d_2_curr_goal < goal_reach_threshold:
           # print('Goal reached!!', d_2_curr_goal)
            self.goal_index = self.goal_index + 1
        if self.goal_index == self.ref_path.shape[0]:
            self.goal_index = 0

    def step(self, action):

        u_t0_H = None
        u_t0_R = None
        u_t0   = None

        x_g = np.zeros((2,1))
            
        # Estimate the human action based on the model
        
        B_hat = copy.deepcopy(self.B_t)
        eps = 0.05
        # eps = 0.0
        B_hat[1,0] = self.mental_state[0,0] + eps

        u_t0_H = get_LQR_control(self.physical_state, self.A_t, B_hat,\
                                self.Q_t, self.R_t,x_g)
        if self.stochastic_human:
            stdev = 0.01 # prev=0.00000001
            u_t0_H = u_t0_H + \
            np.random.multivariate_normal(np.array([0]), stdev * np.array([[1]])).reshape(1,1)
        # make the human stochastic by sampling actions
        
        u_t0_opt = get_LQR_control(self.physical_state, self.A_t, self.B_t,\
                                self.Q_t, self.R_t, x_g)

        if self.noisy_human:
            inst_action_set = build_human_action_set(l_human, 36)
            _,_,u_t0_H_noise,_,sampled_action_prob = \
            get_LQR_control_sample(self.physical_state, self.A_t, B_hat,\
                                  self.Q_t, self.R_t, x_g,\
                                  u_drift_hat, inst_action_set)
            u_t0_H = u_t0_H_noise
            u_t0_opt,_,_,_,_ = \
            get_LQR_control_sample(self.physical_state, self.A_t, self.B_t,\
                                self.Q_t, self.R_t, x_g,\
                                self.u_drift_t, inst_action_set)
        # If the state is near the goal, disable the robot control

        if self.robot_mode == 'passive_teaching':
            u_t0_R = 1.0
        if self.robot_mode == 'active_teaching':
            u_t0_R = self.robot_action_set[action]
        if self.robot_mode == 'random':
            u_t0_R = 1
        u_t0 = u_t0_H * u_t0_R
        
        if self.robot_mode == 'random':
            u_t0 = u_t0_H + \
            np.random.multivariate_normal(np.array([0.01]), 0.01 * np.array([[1]])).reshape(1,1) 

        x_t1 = np.matmul(self.A_t, self.physical_state) \
            + np.matmul(self.B_t, u_t0)
       
        
        # save the data
        self.current_demo_state_traj.append([self.physical_state])
        self.current_demo_human_action_traj.append([u_t0_H])
        if self.step_count >= 2:
            self.current_demo_human_obs_traj[-1][0] = copy.deepcopy(self.physical_state)

        self.current_demo_human_obs_traj.append([x_t1])
        self.current_demo_robot_action_traj.append([u_t0])
        self.current_demo_human_mental_state_traj.append(copy.deepcopy(self.mental_state))
        self.xg_demo_hist.append(x_g)
        self.current_demo_opt_action_traj.append([u_t0_opt])
       
        self.current_demo_human_action_opt_traj.append(np.linalg.norm(u_t0_opt - u_t0_H) )
       
        reward = 0
        if not self.real_human_mode:

            # update the mental state
            if self.human_type == 'use_nn_human' and self.is_updating_internal_model: 
                current_demo_state_traj_copy = copy.deepcopy(self.current_demo_state_traj)
                current_demo_human_action_traj_copy = copy.deepcopy(self.current_demo_human_action_traj)
                current_demo_human_obs_traj_copy = copy.deepcopy(self.current_demo_human_obs_traj)
                #print(current_demo_state_traj_copy)
                # TODO change internalmodelpred
                f_hat_batch_pred = InternalModelPred(self.human_internal_model, 
                                                    current_demo_state_traj_copy,
                                                    current_demo_human_action_traj_copy,
                                                    current_demo_human_obs_traj_copy)
                #print(f_hat_batch_pred)
                self.mental_state[0,0] = f_hat_batch_pred[0,-1,0]
            if self.human_type == 'use_model_human'and self.is_updating_internal_model:
                # use the ground truth model to test the RL algorithm
                A_int_t0, B_int_t0 = update_human_internal_dynamics_model([self.physical_state], \
                                                                [x_t1], [u_t0_H],\
                                                                self.A_t, B_hat,\
                                                                'gradient_decent_threshold', self.human_lr)
                #print(u_drift_int_t0)
                self.mental_state[0,0] = copy.deepcopy(B_int_t0[1,0])
                #print(w_g_int_t0)
            # Calculate reward
            if self.robot_mode == 'active_teaching' or \
            self.robot_mode == 'passive_teaching'or \
            self.robot_mode == 'random':
                
                
                mental_model_error_B = np.abs(self.mental_state[0,0] - self.B_t[1,0])
               
                action_weight = 0
                if mental_model_error_B < 0.02:
                  action_weight = 10.0

                action_cost = np.linalg.norm(u_t0 - u_t0_H)
                reward =  -mental_model_error_B - action_weight * action_cost
            

            self.current_demo_reward_traj.append(reward)
            self.current_demo_task_reward_traj.append(\
                                                    -get_LQR_cost(self.physical_state,\
                                                                    self.A_t, self.B_t,\
                                                                    self.Q_t, self.R_t,\
                                                                    x_g,\
                                                                    u_t0)[0,0])
            self.current_demo_action_reward_traj.append(action_cost)

        # check if simulation ends
        self.step_count += 1
        if self.step_count == self.episode_length:
            done = 1
        else:
            done = 0
        info = {}

        # Return step information
        # update the physical_state
        self.physical_state = x_t1
        self.state = \
        np.concatenate((self.physical_state, self.mental_state), 0).reshape(self.physical_state.shape[0] + self.mental_state.shape[0],)
        return self.state, reward, done, info
    
    def reset(self):
        #print('Env reset!!')
        dx = np.random.rand() * 0.2 - 0.1
        dy = np.random.rand() * 0.2 - 0.1
        #dx = 0
        #dy = 0
        self.physical_state = np.array([[0.4 + dx], [0.0]])
        self.mental_state   = np.array([[1.0]])
        self.state = \
        np.concatenate((self.physical_state, self.mental_state), 0).reshape(self.physical_state.shape[0] + self.mental_state.shape[0],)
        self.step_count = 1
        self.goal_index = 1
        self.current_demo_state_traj = []
        self.current_demo_human_action_traj = []
        self.current_demo_human_obs_traj = []
        self.current_demo_robot_action_traj = []
        self.current_demo_human_mental_state_traj = []
        self.current_demo_reward_traj = []
        self.current_demo_task_reward_traj = []
        self.current_demo_action_reward_traj = []
        self.xg_demo_hist = []
        self.xg_demo_index_hist = []
        self.current_demo_opt_action_traj = []
        self.current_demo_human_action_opt_traj = []

        return self.state
