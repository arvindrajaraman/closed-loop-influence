import numpy as np

# @title Build human fix speed action set
def build_human_action_set(d_l, n_actions):
    theta_set = np.delete(np.linspace(0, 2*np.pi, n_actions+1),-1)
    theta_set = np.reshape(theta_set, (n_actions,1))
    d_x_scaled_set = np.cos(theta_set) * d_l
    d_y_scaled_set = np.sin(theta_set) * d_l
    d_z_scaled_set = 0 * d_y_scaled_set
    action_set = np.concatenate((d_x_scaled_set, d_y_scaled_set, d_z_scaled_set),1)
    return action_set
