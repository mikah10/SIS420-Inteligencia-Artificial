# envs/robotic_env.py
import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class RoboticArmEnv(gym.Env):
    def __init__(self, render=True):
        super().__init__()
        self.render = render
        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Acción: movimiento angular (por ejemplo, 3 articulaciones)
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)

        # Observación: posiciones articulares + vaso (6 valores)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.max_steps = 100
        self.step_counter = 0

        self.robot_id = None
        self.vaso_id = None

        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(os.path.join("assets", "robot.urdf"), useFixedBase=True)
        self.vaso_id = p.loadURDF("cube_small.urdf", basePosition=self.random_vaso_position())

        self.step_counter = 0

        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(3)]
        vaso_pos, _ = p.getBasePositionAndOrientation(self.vaso_id)
        return np.array(joint_states + list(vaso_pos), dtype=np.float32)

    def random_vaso_position(self):
        return [np.random.uniform(0.2, 0.6), 0, 0.05]

    def step(self, action):
        for i in range(3):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[i],
                force=50
            )
        p.stepSimulation()

        self.step_counter += 1
        obs = self._get_obs()
        reward, done = self._compute_reward_done(obs)

        return obs, reward, done, False, {}

    def _compute_reward_done(self, obs):
        vaso_pos = obs[3:]
        distance = np.linalg.norm(np.array(vaso_pos) - np.array([0.4, 0, 0.05]))
        reward = -distance
        done = False

        if distance < 0.05:
            reward += 10
            done = True
        elif self.step_counter >= self.max_steps:
            done = True
        return reward, done

    def render(self):
        pass

    def close(self):
        p.disconnect()
