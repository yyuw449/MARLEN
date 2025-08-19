from onpolicy.envs.my_gym.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from onpolicy.envs.my_gym.envs.mujoco.ant import AntEnv
from onpolicy.envs.my_gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from onpolicy.envs.my_gym.envs.mujoco.hopper import HopperEnv
from onpolicy.envs.my_gym.envs.mujoco.walker2d import Walker2dEnv
from onpolicy.envs.my_gym.envs.mujoco.humanoid import HumanoidEnv
from onpolicy.envs.my_gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from onpolicy.envs.my_gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from onpolicy.envs.my_gym.envs.mujoco.reacher import ReacherEnv
from onpolicy.envs.my_gym.envs.mujoco.swimmer import SwimmerEnv
from onpolicy.envs.my_gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from onpolicy.envs.my_gym.envs.mujoco.pusher import PusherEnv
from onpolicy.envs.my_gym.envs.mujoco.thrower import ThrowerEnv
from onpolicy.envs.my_gym.envs.mujoco.striker import StrikerEnv
