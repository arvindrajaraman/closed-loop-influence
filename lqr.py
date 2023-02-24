import numpy as np
from numpy.random import choice

from dare import get_P

#@title LQR util functions
def get_LQR_control(x_t0_np, A_np, B_np, Q_np, R_np, x_g_np):
    """ Compute the optimal LQR control"""
    P_np = get_P(A_np, B_np, Q_np, R_np).numpy()

    temp = R_np + np.matmul(np.matmul(B_np.transpose(), P_np), B_np)

    K = np.matmul(np.matmul(np.linalg.inv(temp), B_np.transpose()), P_np)
    #print(K)
    return np.matmul(-K, x_t0_np -x_g_np)

def get_LQR_cost(x_t0_np, A_np, B_np, Q_np, R_np, x_g_np, u_t0_np):
  """ Compute the LQR cost for the input state and action, i.e., Q(s,a)"""
  return np.matmul(np.matmul((x_t0_np - x_g_np).T, Q_np), x_t0_np - x_g_np) + np.matmul(np.matmul(u_t0_np.T, R_np), u_t0_np)


#### Does this function sample noisily optimally? ####
def get_LQR_control_sample(x_t0_np, A_np, B_np, Q_np, R_np, x_g_np, u_drift_np,\
                           action_set):
    """ Sample the a LQR control from the input action set"""
    # compute the cost to goal set
    P_n = get_P(A_np, B_np, Q_np, R_np).numpy()
    cost_vector = []
    for i in range(n_actions):
        u = action_set[i,:].reshape(3,1)
        #u = u - np.sign(u) * u_drift_np
        x_next = np.matmul(A_np, x_t0_np) + np.matmul(B, u)
        cost_to_gaol = np.matmul(np.matmul((x_next - x_g_np).T, P_n), (x_next - x_g_np)) + \
                       np.matmul(np.matmul(u.T, R_np), (u))
        cost_vector.append(cost_to_gaol)    
    temp = np.exp(4000 * -np.array(cost_vector))
    action_prob = temp / sum(temp)
    
    opt_action_index = np.argmin(np.array(cost_vector), axis=0)
    sampled_action_index = choice([*range(0, action_set.shape[0], 1)], 1, p=action_prob.squeeze().tolist())
    sampled_action_index = sampled_action_index[0]

    opt_action = action_set[opt_action_index,:].reshape(3,1)
    opt_action = opt_action + np.sign(opt_action) * u_drift_np
    noisy_action = action_set[sampled_action_index,:].reshape(3,1)
    noisy_action = noisy_action + np.sign(noisy_action) * u_drift_np
    #print(sampled_action_index)
    sampled_action_prob = action_prob[sampled_action_index]
    
    return opt_action, opt_action_index, noisy_action, sampled_action_index, sampled_action_prob
