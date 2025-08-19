from harl.envs.gym_custom.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from harl.envs.gym_custom.envs.mujoco.ant import AntEnv
from harl.envs.gym_custom.envs.mujoco.half_cheetah import HalfCheetahEnv
from harl.envs.gym_custom.envs.mujoco.hopper import HopperEnv
from harl.envs.gym_custom.envs.mujoco.walker2d import Walker2dEnv
from harl.envs.gym_custom.envs.mujoco.humanoid import HumanoidEnv
from harl.envs.gym_custom.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from harl.envs.gym_custom.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from harl.envs.gym_custom.envs.mujoco.reacher import ReacherEnv
from harl.envs.gym_custom.envs.mujoco.swimmer import SwimmerEnv
from harl.envs.gym_custom.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from harl.envs.gym_custom.envs.mujoco.pusher import PusherEnv
from harl.envs.gym_custom.envs.mujoco.thrower import ThrowerEnv
from harl.envs.gym_custom.envs.mujoco.striker import StrikerEnv
