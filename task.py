import numpy as np
from physics_sim import PhysicsSim

pi2 = (np.pi / 2.0)


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def abs_arctan_inv(value):
    return (pi2 - abs(np.arctan(value))) / pi2


def abs_arctan(value):
    return (abs(np.arctan(value))) / pi2


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        return self.everything_considered()

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = False
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    # rewards ...
    def reward_speed(self):
        remain_s = self.target_pos - self.sim.pose[:3]
        target_s = remain_s / (self.sim.runtime - self.sim.time)
        delta_s = abs(self.sim.v - target_s).sum()
        return 2 - sigmoid(delta_s)

    def reward_basic(self):
        return np.tanh(1 - 0.003 * (abs(self.sim.pose[:3] - self.target_pos))).sum()

    def everything_considered(self):
        euler = self.sim.pose[3:6]
        current_pos = self.sim.pose[:3]
        start_pos = self.sim.init_pose[:3]
        speed = self.sim.v
        s_max = speed[2]  # speed in z
        s_min = speed[:2]  # speed in x, y
        angular_speed = self.sim.angular_v
        delta_position = current_pos - start_pos
        dp_max = delta_position[2]  # max this (Z)
        dp_min = delta_position[:2]  # min this (X, Y)

        euler = (abs_arctan(euler - np.pi) * 1.25).sum() / 3  # min = 0, max = 1
        angular_speed = abs_arctan_inv(angular_speed).sum() / 3  # min = 0, max = 1
        dp_min = abs_arctan_inv(dp_min).sum() / 2  # min = 0, max = 1s
        dp_max = abs_arctan(dp_max)  # min = 0, max = 1
        s_min = abs_arctan_inv(s_min).sum() / 2  # min = 0, max = 1
        s_max = np.arctan(s_max) / pi2  # min = -1 , max = 1

        sum = euler + angular_speed + dp_max + dp_min + s_max + s_min
        mx = 6
        mn = -1
        delta = mx - mn
        reward = (sum + abs(mn)) / delta
        return reward
