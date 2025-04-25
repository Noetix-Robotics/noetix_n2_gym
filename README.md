## Installation

1. Install Isaac Gym:
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - Run an example with `cd examples && python 1080_balls_of_solitude.py`.
   - Consult `isaacgym/docs/index.html` for troubleshooting.
2. Install noetix_rl_gym:
   - Clone this repository.
   - `cd noetix_rl_gym && pip install -e .`



## Usage Guide

#### Examples

```bash
# Launching PPO Policy Training for 'v1' Across 4096 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=n2 --run_name v1 --headless --num_envs 4096

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=n2 --run_name v1

```

#### 1. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=n2 --load_run log_file_path --name run_name
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=n2 --load_run log_file_path --name run_name
  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.

#### 2. Sim-to-sim
- **Please note: Before initiating the sim-to-sim process, ensure that you run `play.py` to export a JIT policy.**
- **Mujoco-based Sim2Sim Deployment**: Utilize Mujoco for executing simulation-to-simulation (sim2sim) deployments with the command below:
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```


#### 3. Parameters
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

#### 4. Command-Line Arguments
For RL training, please refer to `humanoid/utils/helpers.py#L161`.
For the sim-to-sim process, please refer to `humanoid/scripts/sim2sim.py#L169`.

## Code Structure

1. Every environment hinges on an `env` file (`legged_robot.py`) and a `configuration` file (`legged_robot_config.py`). The latter houses two classes: `LeggedRobotCfg` (encompassing all environmental parameters) and `LeggedRobotCfgPPO` (denoting all training parameters).
2. Both `env` and `config` classes use inheritance.
3. Non-zero reward scales specified in `cfg` contribute a function of the corresponding name to the sum-total reward.
4. Tasks must be registered with `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration may occur within `envs/__init__.py`, or outside of this repository.


## Add a new environment 

The base environment `legged_robot` constructs a rough terrain locomotion task. The corresponding configuration does not specify a robot asset (URDF/ MJCF) and no reward scales.

1. If you need to add a new environment, create a new folder in the `envs/` directory with a configuration file named `<your_env>_config.py`. The new configuration should inherit from existing environment configurations.
2. If proposing a new robot:
    - Insert the corresponding assets in the `resources/` folder.
    - In the `cfg` file, set the path to the asset, define body names, default_joint_positions, and PD gains. Specify the desired `train_cfg` and the environment's name (python class).
    - In the `train_cfg`, set the `experiment_name` and `run_name`.
3. If needed, create your environment in `<your_env>.py`. Inherit from existing environments, override desired functions and/or add your reward functions.
4. Register your environment in `humanoid/envs/__init__.py`.
5. Modify or tune other parameters in your `cfg` or `cfg_train` as per requirements. To remove the reward, set its scale to zero. Avoid modifying the parameters of other environments!
6. If you want a new robot/environment to perform sim2sim, you may need to modify `humanoid/scripts/sim2sim.py`: 
    - Check the joint mapping of the robot between MJCF and URDF.
    - Change the initial joint position of the robot according to your trained policy.

## Troubleshooting

Observe the following cases:

```bash
# error
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# solution
# set the correct path
export LD_LIBRARY_PATH="~/miniconda3/envs/your_env/lib:$LD_LIBRARY_PATH" 

# OR
sudo apt install libpython3.8

# error
AttributeError: module 'distutils' has no attribute 'version'

# solution
# install pytorch 1.12.0
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# error, results from libstdc++ version distributed with conda differing from the one used on your system to build Isaac Gym
ImportError: /home/roboterax/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20` not found (required by /home/roboterax/carbgym/python/isaacgym/_bindings/linux64/gym_36.so)

# solution
mkdir ${YOUR_CONDA_ENV}/lib/_unused
mv ${YOUR_CONDA_ENV}/lib/libstdc++* ${YOUR_CONDA_ENV}/lib/_unused
```

