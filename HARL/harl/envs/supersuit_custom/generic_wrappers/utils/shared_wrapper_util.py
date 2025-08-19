import functools
import gymnasium
from harl.envs.pettingzoo_custom.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap
from harl.envs.supersuit_custom.utils.wrapper_chooser import WrapperChooser
from harl.envs.pettingzoo_custom.utils import BaseParallelWraper


class shared_wrapper_aec(PettingzooWrap):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifier_class = modifier_class
        self.modifiers = {}
        self._cur_seed = None
        self._cur_options = None

        if hasattr(self.env, "possible_agents"):
            self.add_modifiers(self.env.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list):
        for agent in agents_list:
            if agent not in self.modifiers:
                # populate modifier spaces
                self.modifiers[agent] = self.modifier_class()
                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].reset(
                    seed=self._cur_seed, options=self._cur_options
                )

                # modifiers for each agent has a different seed
                if self._cur_seed is not None:
                    self._cur_seed += 1

    def reset(self, seed=None, return_info=False, options=None):
        self._cur_seed = seed
        self._cur_options = options

        for mod in self.modifiers.values():
            mod.reset(seed=seed, options=options)
        super().reset(seed=seed, options=options)

        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(
            super().observe(self.agent_selection)
        )

    def step(self, action):
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            action = None
        super().step(action)
        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(
            super().observe(self.agent_selection)
        )

    def observe(self, agent):
        return self.modifiers[agent].get_last_obs()


class shared_wrapper_parr(BaseParallelWraper):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifier_class = modifier_class
        self.modifiers = {}
        self._cur_seed = None
        self._cur_options = None

        if hasattr(self.env, "possible_agents"):
            self.add_modifiers(self.env.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list):
        for agent in agents_list:
            if agent not in self.modifiers:
                # populate modifier spaces
                self.modifiers[agent] = self.modifier_class()
                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].reset(
                    seed=self._cur_seed, options=self._cur_options
                )

                # modifiers for each agent has a different seed
                if self._cur_seed is not None:
                    self._cur_seed += 1

    def reset(self, seed=None, options=None):
        self._cur_seed = seed
        self._cur_options = options

        observations = super().reset(seed=seed, options=options)
        self.add_modifiers(self.agents)
        for agent, mod in self.modifiers.items():
            mod.reset(seed=seed, options=options)
        observations = {
            agent: self.modifiers[agent].modify_obs(obs)
            for agent, obs in observations.items()
        }
        return observations

    def step(self, actions):
        actions = {
            agent: self.modifiers[agent].modify_action(action)
            for agent, action in actions.items()
        }
        observations, rewards, terminations, truncations, infos = super().step(actions)
        self.add_modifiers(self.agents)
        observations = {
            agent: self.modifiers[agent].modify_obs(obs)
            for agent, obs in observations.items()
        }
        return observations, rewards, terminations, truncations, infos


class shared_wrapper_gym(gymnasium.Wrapper):
    def __init__(self, env, modifier_class):
        super().__init__(env)
        self.modifier = modifier_class()
        self.observation_space = self.modifier.modify_obs_space(self.observation_space)
        self.action_space = self.modifier.modify_action_space(self.action_space)

    def reset(self, seed=None, options=None):
        self.modifier.reset(seed=seed, options=options)
        obs = super().reset(seed=seed, options=options)
        obs = self.modifier.modify_obs(obs)
        return obs

    def step(self, action):
        obs, rew, term, trunc, info = super().step(self.modifier.modify_action(action))
        obs = self.modifier.modify_obs(obs)
        return obs, rew, term, trunc, info


shared_wrapper = WrapperChooser(
    aec_wrapper=shared_wrapper_aec,
    gym_wrapper=shared_wrapper_gym,
    parallel_wrapper=shared_wrapper_parr,
)
