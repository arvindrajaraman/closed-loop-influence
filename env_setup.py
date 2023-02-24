import numpy as np
import torch

from dare import get_P
from device import device

# @title  Environment dynamics, robot reward function

# Time integral
dt = 0.2

# Robot dynamics matrix
A        = [[1., dt], [0., 1.]]
A        = np.array(A)
A_tensor = torch.from_numpy(A).to(device).double()
nX       = A.shape[0]

# Robot control matrix
B        = [[0.], [0.5]]
B        = np.array(B)
B_tensor = torch.from_numpy(B).to(device).double()
nU       = B.shape[1]

# Optimal LQR cost function (Q and R)
Q        = [[1.0, 0.], [0., 1.]]
Q        = np.array(Q)
Q_tensor = torch.from_numpy(Q).to(device).double()
R        = 5 * np.eye(1)
R_tensor = torch.from_numpy(R).to(device).double()
R_inv    = np.linalg.inv(R)

# Human's internal robot dynamics
A_int = A
A_int = np.array(A_int)

# Note that here the human has a wrong estimate about the B matrix
B_int = [[0.], [0.1]]
B_int = np.array(B_int)

# Compute human's LQR DARE solution (i.e., human's believed solution matrix)
P_int = get_P(A_int, B_int, Q, R)
# print("Human internal control matrix:", P_int)
# print()
# Compute robot's LQR DARE solution (i.e., the optimal solution matrix)
# P_true = get_P(A, B, Q, R)
# print("Optimal control matrix:", P_true)

# Mask for the dynamics gradient update
A_grad_mask = np.array(A_int != A) * 1.0
B_grad_mask = np.array(B_int != B) * 1.0
# print(A_grad_mask)
# print(B_grad_mask)

u_t0_R_aug_set  = [1., 0.2, 0.6, 1.2 ,1.8]
