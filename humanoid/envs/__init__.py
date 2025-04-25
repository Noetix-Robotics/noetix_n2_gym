from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from humanoid.utils.task_registry import task_registry

from .n2.n2_env import N2Env
from .n2.n2_config import N2RoughCfg, N2RoughCfgPPO

task_registry.register( "n2", N2Env, N2RoughCfg(), N2RoughCfgPPO() )









