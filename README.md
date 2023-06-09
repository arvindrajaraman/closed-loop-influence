# Closed Loop Influence

In a human-robot interaction setting, humans inevitably have flawed or biased understanding of the environment around them. They may not be sure about extrinsic factors of the world (e.g. a robot arm's joint friction), or their own internal goals/preferences (e.g. how the robot's arm should be configured). Humans often have a long learning curve when learning to operate a robotic arm with precision and dexterity. What if we could have the robot influence the human, with the human-in-the-loop, and as a result, speed up this learning process?

Our work, Towards Personalized Robotic Influence of Human Internal State, sets up a two-phase process (estimation and influence), where estimation involves learning about the human's flawed understanding of reality and influence involves the robot using RL 

Here is a high-level overview of what each file does:
- `build_human_action_set.py` - generates the set of all possible actions a human can take in our discretized action space.
- `dare.py` - the math behind the Discrete Algebraic Riccati Equation (DARE), which underlies our optimal control.
- `data_gen.py` - generates synthetic state-action trajectories of a human interacting with a robot.
- `device.py` - sets whether GPU/CPU should be used for all components of the project.
- `env_setup.py` - sets the values of our control matrices.
- `estimation.py` - performs the entire estimation process.
- `human.py` - outlines the human/robot environment for OpenAI Gym.
- `lqr.py` - uses `dare.py` to derive optimal controls that the human would take.
- `models.py` - defines different models that can be used in the estimation process.
- `train_mlp.ipynb` - interactive environment to train a vanilla neural network-based estimator of the human's internal state.
- `train_transformer.ipynb` - interactive environment to train a Transformer-based estimator of the human's internal state.
