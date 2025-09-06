# MARLEN: Multi-Agent Reinforcement Learning with Exogenous Non-stationarity

## Environment Setup
Please refer [HARL](https://github.com/PKU-MARL/HARL), [A2PO](https://github.com/xihuai18/A2PO-ICLR2023), [MAZero](https://github.com/liuqh16/MAZero), [Alignment](https://github.com/StanfordVL/alignment), [MARR](https://github.com/CNDOTA/ICML24-MARR/tree/main/marr-mpe) and [DPO](https://github.com/PKU-RL/DPO) for environmental setup.

---

## Reproducing results/plots in the paper

### HARL
To reproduce halfCheetah env on happo algorithm with both gravity and reward non-stationarity run : (change the algorithm as needed)
```bash
python examples/train.py --load_config HARL/tuned_configs/mamujoco/HalfCheetah-v2-2x3/happo/config.json --exp_name mujoco_half_cheetah_happo_non_stat
```
To reproduce ant env on happo algorithm with gravity non-stationarity run : (change the algorithm as needed)
```bash
python examples/train.py --load_config /hdd3/marl_new/HARL/tuned_configs/mamujoco/Ant-v2-4x2/happo/config.json --exp_name mujoco_ant_happo_non_stat
```
To reproduce walker env on happo algorithm with reward non-stationarity run : (change the algorithm as needed)
```bash
python examples/train.py --load_config /hdd3/marl_new/HARL/tuned_configs/mamujoco/Walker2d-v2-2x3/happo/config.json --exp_name mujoco_walker_happo_non_stat
```
To reproduce simple spread (continuous) env on happo algorithm with landmark non-stationarity run : (change the algorithm as needed)
```bash
python examples/train.py --load_config HARL/tuned_configs/pettingzoo_mpe/simple_spread_v2-continuous/happo/config.json --exp_name simple_spread_continuous_happo_non_stat
```
To reproduce simple spread (discrete) env on happo algorithm with landmark non-stationarity run : (change the algorithm as needed)
```bash
python examples/train.py --load_config HARL/tuned_configs/pettingzoo_mpe/simple_spread_v2-discrete/happo/config.json --exp_name simple_spread_discrete_happo_non_stat
```


Use the configs present in the HARL/tuned_configs folder

### DPO
To reproduce simple spread on DPO algorithm: 
```bash
python train_mpe.py --use_eval --inner_refine True --penalty_method True --dtar_kl 0.01 --experiment_name spread_nns_2 --num_env_steps 1000000 --group_name dpo --seed 2 --multi_rollout True --n_rollout_threads 1 --map_name simple_spread --num_agents 3 --num_landmarks 3
```

### MARR
```bash
python src/main.py --config=rrfacmac_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_6a_NS t_max=2000000
```

### MAZero
```bash
python main.py --opr train_sync --case $env --env_name $scenerio --exp_name $exp_name --seed $seed \
    --num_cpus 64 --num_gpus 1 --train_on_gpu --reanalyze_on_gpu --selfplay_on_gpu \
    --data_actors 1 --num_pmcts 4 --reanalyze_actors 30 \
    --test_interval 1000 --test_episodes 32 --target_model_interval 200 \
    --batch_size 256 --num_simulations $N --sampled_action_times $K \
    --training_steps 14000 --last_step 1000 --lr 5e-4 --lr_adjust_func const --max_grad_norm 5 \
    --total_transitions 28000 --start_transition 500 --discount 0.99 \
    --target_value_type pred-re --revisit_policy_search_rate 1.0 --use_off_correction \
    --value_transform_type vector --use_mcts_test \
    --use_priority --use_max_priority \
    --PG_type $PG_type --awac_lambda $awac_lambda --adv_clip $adv_clip \
    --mcts_rho $mcts_rho --mcts_lambda $mcts_lambda 
```

### Alignment
```bash
# Training
python map/train_multi_sacd.py --task simple_spread_in_NS_change_landmark_with_episode --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 5 --save-models --benchmark  --logdir log/simple_spread

# Testing
python map/evaluate_multi_sacd.py --savedir result --logdir log/simple_spread
```

Simple Spread + Alignment without non-stationarity:    
<video width="600" controls>
  <source src="Alignment_videos/NO_NS.gif" type="video/mp4">
</video>

Simple Spread + Alignment with reward non-stationarity (varying landmarks):     
<video width="600" controls>
  <source src="Alignment_videos/NS.gif" type="video/mp4">
</video>



### A2PO
```bash
bash run_scripts/MUJOCO/walker2d.sh
```

---

## Implementing MARLEN in custom MARL algorithms
### Varying transition dynamics through gravity
Add the following gravity function to the reset_model() method of your algorithm. Declare a variable '_meta_time' to track the time 
```bash
def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        # GRAVITY NON-STATIONARITY: Gravity changes over time based on meta_time
        self.sim.model.opt.gravity[-1] -= 9 * (1.1 ** (-math.ceil(self._meta_time/300))) * math.sin(math.pi * 0.01 * self._meta_time)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self._meta_time += 1
        return self._get_obs()
```
### Episode-varying reward function in Multi-Agent MuJoCo:
Add the following direction reversal to the reset_model() method of your algorithm
```bash
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
        # DIRECTIONAL REVERSAL NON-STATIONARITY: Reward direction changes based on _mujoco_dir
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
def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        # DIRECTIONAL REVERSAL: Direction multiplier changes sign each reset
        self._mujoco_dir *= -1
        self._meta_time += 1
        return self._get_obs()
```

### Episode-varying reward function in Simple Spread
Reset variables
```bash
def reset_world(self, world):
        # 3 agents 3 landmarks
        if self.iter_num != 2:
            self.iter_num += 1
        else:
            self.iter_num = 0
```
Modify the global_reward() method to add landmark assignment
```bash
def global_reward(self, world):
        rew = 0
        _rew = 0
        for i, lm in enumerate(world.landmarks):
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            if iter_num == 0:
                map_d = {0: 0, 1: 1, 2: 2}
                min_dist, min_idx = self.get_min_index(dists)
                if i != min_idx:
                    _rew = min_dist + 2
                else:
                    _rew = min_dist
            elif iter_num == 1:
                map_d = {0: 2, 1: 0, 2: 1}
                min_dist, min_idx = self.get_min_index(dists)
                if map_d[i] != min_idx:
                    _rew = min_dist + 2
                else:
                    _rew = min_dist
            elif iter_num == 2:
                map_d = {0: 1, 1: 2, 2: 0}
                min_dist, min_idx = self.get_min_index(dists)
                if map_d[i] != min_idx:
                    _rew = min_dist + 2
                else:
                    _rew = min_dist
            elif iter_num == -1:
                min_dist, _ = self.get_min_index(dists)
                _rew = min_dist
            rew -= _rew
        return rew
```

### Observation space Non-stationarity
#### Time-varying Observation space in Predator Prey
Declare the noise patterns
```bash
self.up = np.linspace(0.1, 0.01, num=480)
self.down = np.linspace(0.01, 0.1, num=480)
self.iter = -1
self.iter_num = 0
```
Add noise to position and velocity in the observation() function 
```bash
var = np.random.normal(
                0, 
                self.up[self.iter_num] if self.iter == 1 else self.down[self.iter_num]
                )
noisy_x_y_pos_other[0] += var
noisy_x_y_pos_other[1] += var

var = np.random.normal(
                    0, 
                    self.up[self.iter_num] if self.iter == 1 else self.down[self.iter_num]
                    )
noisy_x_y_vel_other[0] += var
noisy_x_y_vel_other[1] += var
self.iter_num += 1
```
Reset variables
```bash
def reset_world(self, world):
        self.iter *= -1
        self.iter_num = 0
```

#### Time-varying observation space in SMAC: 
Add noise to the enemy and ally positions in get_obs_agent() function
```bash
var =  np.random.normal(0,
                1.0 if self.iter == 1 else 2.0
                )
e_x += var
e_y += var

al_x += var
al_y += var
```

Reset variables
```bash
def reset(self):
        self.iter *= -1
```
---

## Acknowledgement
We thank the authors of the following repositories:<br/>
https://github.com/PKU-MARL/HARL<br/>
https://github.com/xihuai18/A2PO-ICLR2023<br/>
https://github.com/liuqh16/MAZero<br/>
https://github.com/StanfordVL/alignment<br/>
https://github.com/CNDOTA/ICML24-MARR<br/>
https://github.com/PKU-RL/DPO<br/>
