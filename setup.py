from setuptools import find_packages
from distutils.core import setup

setup(
    name='humanoid',
    version='1.0.0',
    author='Huaxing Huang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='huaxinghuang@noetixrobotics.com',
    description='Isaac Gym environments for humanoid robot',
    install_requires=['isaacgym',  # preview4
                      'wandb',
                      'tensorboard',
                      'tqdm',
                      'onnx',
                      'pynput',
                      'numpy==1.23.5',
                      'opencv-python',
                      'mujoco==2.3.6',
                      'pybullet==3.2.6',
                      'mujoco-python-viewer',
                      'matplotlib',
                      ]
)
