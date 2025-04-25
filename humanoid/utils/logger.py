import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, env):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.env = env
        self.dt = self.env.dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        target=self._plot()
        # self.plot_process = Process(target=self._plot)
        # self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 6

        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        plt.figure(figsize=(18, 10))
        for i in range(nb_rows):
            for j in range(nb_cols):
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                if log["target_" + self.env.dof_names[i*nb_cols+j]]: 
                    plt.plot(time, log["target_" + self.env.dof_names[i*nb_cols+j]], label="Desired angle", color="blue")
                if log["pos_" + self.env.dof_names[i*nb_cols+j]]:
                    plt.plot(time, log["pos_" + self.env.dof_names[i*nb_cols+j]], label="measured", color="red", linestyle="--")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual angle (rad)")
                plt.title(self.env.dof_names[i*nb_cols+j])
                plt.grid(True)
        plt.tight_layout()
        plt.savefig('pd_dof_pos.png')

        plt.figure(figsize=(18, 10))
        for i in range(nb_rows):
            for j in range(nb_cols):
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                if log["vel_" + self.env.dof_names[i*nb_cols+j]]: 
                    plt.plot(time, log["vel_" + self.env.dof_names[i*nb_cols+j]], label="measured", color="black")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual vel (rad/s)")
                plt.title(self.env.dof_names[i*nb_cols+j])
                plt.grid(True)
        plt.tight_layout()
        plt.savefig('pd_dof_vel.png')

        plt.figure(figsize=(18, 10))
        for i in range(nb_rows):
            for j in range(nb_cols):
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                if log[self.env.dof_names[i*nb_cols+j]]: 
                    plt.plot(time, log[self.env.dof_names[i*nb_cols+j]], label="measured", color="green")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual torque (rad)")
                plt.title(self.env.dof_names[i*nb_cols+j])
                plt.grid(True)
        plt.tight_layout()
        plt.savefig('pd_torques.png')

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()