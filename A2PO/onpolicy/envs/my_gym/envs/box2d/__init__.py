try:
    import Box2D
    from onpolicy.envs.my_gym.envs.box2d.lunar_lander import LunarLander
    from onpolicy.envs.my_gym.envs.box2d.lunar_lander import LunarLanderContinuous
    from onpolicy.envs.my_gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from onpolicy.envs.my_gym.envs.box2d.car_racing import CarRacing
except ImportError:
    Box2D = None
