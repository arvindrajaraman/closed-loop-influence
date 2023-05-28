import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import os

from dare import Riccati
from data_gen import generate_simulated_data
from device import device
from env_setup import *
from human import HumanRobotEnv
from models import ThetaEstimatorTransformer

writer = SummaryWriter()

"""SETUP HYPERPARAMETERS"""

# Generate trajectories to train estimator
sim_policy = dict()
sim_policy['human_state'] = 'fixed'  # can be 'fixed' or 'varying'
sim_policy['mental_state'] = 'fixed'  # can be 'fixed' or 'varying'

sim_policy['human_state_init'] = [[0.4], [0.0]] # only needed when human_state = 'fixed'
sim_policy['mental_state_init'] = [[1.0]] # only needed when mental_state = 'fixed'

sim_time = 20
n_demo = 100
is_updating_internal_model = True
stochastic_human = False
human_lr = 2.0

train_split = 0.7
train_size = int(n_demo * train_split)
test_size = n_demo - train_size

epochs = 100
model_lr = 0.01

data = generate_simulated_data(sim_policy, sim_time, n_demo, is_updating_internal_model, stochastic_human, human_lr)
robot_states, human_actions, human_obs, human_mental_states = data

robot_states_train, robot_states_test = robot_states[:train_size], robot_states[train_size:]
human_actions_train, human_actions_test = human_actions[:train_size], human_actions[train_size:]
human_obs_train, human_obs_test = human_obs[:train_size], human_obs[train_size:]
human_mental_states_train, human_mental_states_test = human_mental_states[:train_size], human_mental_states[train_size:]

"""DATA VISUALIZATION"""
# def physical_states_grid():
#     n = 2 if (n_demo < 9) else 3
#     fig, axs = plt.subplots(n, n)
#     for i in range(n):
#         for j in range(n):
#             ax = axs[i][j]
#             idx = (i*n) + j
#             human_traj = np.array(robot_states_train[idx]).squeeze()
#             ax.plot(human_traj[:,0],human_traj[:,1],'bo', markersize=3)
#             ax.axis(xmin=-1, xmax=1, ymin=-1, ymax=1)
#             ax.axis('equal')

#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     writer.add_image('data/physical_states', data, 0)

# physical_states_grid()

# def mental_states_grid():
#     n = 2 if (n_demo < 9) else 3
#     fig, axs = plt.subplots(n, n)
#     for i in range(n):
#         for j in range(n):
#             ax = axs[i][j]
#             idx = (i*n) + j
#             print(human_mental_states_train[idx])
#             human_internal_state_traj = np.array(human_mental_states_train[idx]).squeeze()
#             ax.plot(human_internal_state_traj, 'bo', markersize=3)
    
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     writer.add_image('data/mental_states', data, 0)

# mental_states_grid()

"""PREPROCESSING DATA FOR ESTIMATOR"""
states_train = torch.tensor(robot_states_train, device=device)
states_train = states_train.view(train_size * sim_time, nX)
states_test = torch.tensor(robot_states_test, device=device)
states_test = states_test.view(test_size * sim_time, nX)

actions_train = torch.tensor(human_actions_train, device=device)
actions_train = actions_train.view(train_size * sim_time, nU)
actions_test = torch.tensor(human_actions_test, device=device)
actions_test = actions_test.view(test_size * sim_time, nU)

obs_train = torch.tensor(human_obs_train, device=device)
obs_train = obs_train.view(train_size * sim_time, nX)
obs_test = torch.tensor(human_obs_test, device=device)
obs_test = obs_test.view(test_size * sim_time, nX)

inputs_train = torch.cat((states_train, actions_train, obs_train), axis=1)
inputs_train = inputs_train.view(train_size, sim_time, nX + nU + nX).double()
inputs_test = torch.cat((states_test, actions_test, obs_test), axis=1)
inputs_test = inputs_test.view(test_size, sim_time, nX + nU + nX).double()

print('Train:', inputs_train.shape)
print('Test:', inputs_test.shape)

"""ESTIMATOR TRAINING"""
transformer_estimator = ThetaEstimatorTransformer().to(device).double()
optimizer = torch.optim.Adam(transformer_estimator.parameters(), lr=model_lr)

def predict_action(state, theta_H):
    B_hat_tensor = theta_H * torch.tensor([[0., ],[1.0]], device = device).double()
    P_hat = Riccati.apply(A_tensor, B_hat_tensor, Q_tensor, R_tensor)

    K = torch.linalg.multi_dot((
        torch.linalg.pinv(torch.add(
            R_tensor,
            torch.linalg.multi_dot((torch.transpose(B_hat_tensor, 0, 1), P_hat, B_hat_tensor))
        )),
        torch.transpose(B_hat_tensor, 0, 1),
        P_hat,
        A_tensor
    ))
    action_pred = -torch.matmul(K, state)
    return action_pred

def forward_pass(model, inputs, curr_traj_idx):
    inputs = inputs.reshape(-1, inputs.shape[0], inputs.shape[1])
    theta_Hs = model(inputs)
    theta_Hs = theta_Hs.reshape(sim_time)
    theta_H_error = 0

    step_losses = []
    for i in range(sim_time - 1):
        # theta_H = theta_Hs[i] * 0.0 + human_mental_states[curr_traj_idx][i][0][0]
        theta_H = theta_Hs[i]
        theta_H_true = human_mental_states_train[curr_traj_idx][i][0][0]
        theta_H_error += torch.linalg.norm(theta_H - theta_H_true).data.item()

        input = inputs[0][i]
        state, action, obs = torch.split(input, [2, 1, 2])
        
        action_pred = predict_action(state, theta_H)
        # print(action_pred, action)

        loss_fn = nn.MSELoss()
        loss = loss_fn(action_pred, action)

        if abs(loss) > 20:
            ipdb.set_trace()

        step_losses.append(loss)
    
    return step_losses, theta_H_error

def train_epoch(model, inputs):
    model.train()
    all_losses = []
    theta_H_error_all = 0
    for idx in range(inputs.shape[0]):
        inp = inputs[idx]
        step_losses, theta_H_error = forward_pass(model, inp, idx)
        theta_H_error_all += theta_H_error
        all_losses += step_losses
    
    optimizer.zero_grad()
    total_loss = sum(all_losses) / (sim_time * inputs.shape[0])
    total_loss.backward()
    optimizer.step()

    theta_H_error_all /= (sim_time * inputs.shape[0])

    return total_loss.data.item(), theta_H_error_all

def test_epoch(model, inputs):
    model.eval()
    all_losses = []
    theta_H_error_all = 0
    for idx in range(inputs.shape[0]):
        inp = inputs[idx]
        step_losses, theta_H_error = forward_pass(model, inp, idx)
        theta_H_error_all += theta_H_error
        all_losses += step_losses
    
    total_loss = sum(all_losses) / (sim_time * inputs.shape[0])
    theta_H_error_all /= (sim_time * inputs.shape[0])

    return total_loss.data.item(), theta_H_error_all

epoch_list = range(1, epochs+1)
for epoch in tqdm(epoch_list):
    # print(f'Epoch {epoch}: theta_H_error={theta_H_error}')
    train_loss, theta_H_train_err = train_epoch(transformer_estimator, inputs_train)
    test_loss, theta_H_test_err = test_epoch(transformer_estimator, inputs_test)

    # train_losses.append(train_loss)
    # test_losses.append(test_loss)
    # theta_H_train_errs.append(theta_H_train_err)
    # theta_H_test_errs.append(theta_H_test_err)

    writer.add_scalar('action_pred_err/train', train_loss, epoch)
    writer.add_scalar('action_pred_err/test', test_loss, epoch)
    writer.add_scalar('model_pred_err/train', theta_H_train_err, epoch)
    writer.add_scalar('model_pred_err/test', theta_H_test_err, epoch)
