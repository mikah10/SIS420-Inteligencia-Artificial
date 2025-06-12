# scripts/train.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.robotic_env import RoboticArmEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def main():
    env = DummyVecEnv([lambda: RoboticArmEnv(render=False)])
    check_env(RoboticArmEnv())  # Verifica compatibilidad

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_robotic_arm")

if __name__ == "__main__":
    main()
