import numpy as np
import math
import pickle
from harl.envs.gym_custom import utils
from harl.envs.gym_custom.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._wind_frc = 0.0
        self._meta_time = -1
        self._mujoco_dir = 1
        self._obs_dp = False
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)
        

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.1 * np.square(a).sum()
        contact_cost = (
            0.1 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward * self._mujoco_dir - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        if self._obs_dp:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat,
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                    [
                        0.02 * self._wind_frc,
                        self._mujoco_dir, 
                    ]
                ]
            )
        else:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat,
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                ]
            ).ravel()
            

    def reset_model(self):
        # self.model.opt.wind[:] = np.array([10, 10, 10], dtype=float)
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        self.sim.model.opt.gravity[-1] -= 4 * (1.1 ** (-math.ceil(self._meta_time/300))) * math.sin(math.pi * 0.01 * self._meta_time)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self._mujoco_dir *= -1
        self._meta_time += 1
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
