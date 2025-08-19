import numpy as np
import math
import pickle
from harl.envs.gym_custom import utils
from harl.envs.gym_custom.envs.mujoco import mujoco_env
import inspect
from single_variable import get_global_value


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self._target_vel = 0.0
        self._wind_frc = 0.0
        self._meta_time = -1
        self._mujoco_dir = 1
        self._avg = 1.5
        self._mag = 1.5
        self._dtheta = 0.5
        self.mujoco_dir = 1
        self._obs_dp = False
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)
        
        self.action_space.low[-1] = 0.0
        self.action_space.high[-1] = 0.0

    def step(self, action):
        # action[-1] = self._wind_frc
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        # print(reward_run)
        reward_run *= self._mujoco_dir
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        if self._obs_dp:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    [
                        self._wind_frc, 
                        self._mujoco_dir,
                    ]
                ]
            ).ravel()
        else:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                ]
            ).ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        
        self.sim.model.opt.gravity[-1] -= 9 * (1.1 ** (-math.ceil(self._meta_time/300))) * math.sin(math.pi * 0.01 * self._meta_time)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self._target_vel = self._avg + self._mag * np.sin(
            self._dtheta * self._meta_time
        )
        self._mujoco_dir *= -1
        self._meta_time += 1
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
