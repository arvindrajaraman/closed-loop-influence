import numpy as np
import random
from tqdm import tqdm

from env_setup import *
from human import HumanRobotEnv
from models import ThetaEstimatorTransformer

def gen_physical_state(sim_policy: dict, x1_lo: float, x2_lo: float, x1_hi: float, x2_hi: float):
    if sim_policy['human_state'] == 'fixed':
        assert 'human_state_init' in sim_policy
        return np.array(sim_policy['human_state_init'])
    elif sim_policy['human_state'] == 'varying':
        x1 = random.uniform(x1_lo, x1_hi)
        x2 = random.uniform(x2_lo, x2_hi)
        return np.array([[x1], [x2]])
    else:
        raise ValueError("unknown sim_policy['human_state']")
        
def gen_mental_state(sim_policy: dict):
    if sim_policy['mental_state'] == 'fixed':
        assert 'mental_state_init' in sim_policy
        return np.array(sim_policy['mental_state_init'])
    elif sim_policy['mental_state'] == 'varying':
        x = random.uniform(0.0, 1.0)
        return np.array([[x]])
    else:
        raise ValueError("unknown sim_policy['mental_state']")

def generate_simulated_data(sim_policy: dict, sim_time: int, n_demo: int,
                            is_updating_internal_model: bool, stochastic_human: bool,
                            human_lr: float, influence_type: str):
    # Simulate how human updates the internal model

    robot_states = []
    human_actions = []
    human_obs = []
    human_mental_states = []

    for i in tqdm(range(n_demo)):
        human_env = HumanRobotEnv('passive_teaching', 1.0, 'use_model_human', is_updating_internal_model, human_lr, influence_type)
        human_env.set_environment(A, B, Q, R, None, None, sim_time)
        human_env.set_action_set(None, u_t0_R_aug_set)
        human_env.set_human_internal_model(None)

        x1_lo = human_env.observation_space.low[0]
        x2_lo = human_env.observation_space.low[1]
        x1_hi = human_env.observation_space.high[0]
        x2_hi = human_env.observation_space.high[1]
        
        # Initialize the robot state [0.4; 0], human mental state [1.0] (human thinks B[1,0]=1.0)
        human_env.set_human_state(gen_physical_state(sim_policy, x1_lo, x2_lo, x1_hi, x2_hi),
                                gen_mental_state(sim_policy))
        
        human_env.noisy_human = False
        human_env.stochastic_human = stochastic_human
        for i in range(sim_time):
            human_env.step(None)
        
        robot_states.append(human_env.current_demo_state_traj)
        human_actions.append(human_env.current_demo_human_action_traj)
        human_obs.append(human_env.current_demo_human_obs_traj)
        human_mental_states.append(human_env.current_demo_human_mental_state_traj)

    return robot_states, human_actions, human_obs, human_mental_states
