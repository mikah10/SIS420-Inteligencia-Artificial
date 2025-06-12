import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

# Configuración del entorno
env = gym.make("Taxi-v3", render_mode="human")
# Asegurarse de que el entorno se haya creado correctamente


# Inicialización de la tabla Q con ceros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hiperparámetros
alpha = 0.1       # tasa de aprendizaje
gamma = 0.99      # factor de descuento
epsilon = 0.1     # probabilidad de exploración
num_episodes = 10000
max_steps = 100   # pasos máximos por episodio

# Para gráficas
rewards_all_episodes = []

# Entrenamiento
for episode in range(num_episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # ε-greedy: exploración o explotación
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # exploración
        else:
            action = np.argmax(q_table[state])  # explotación

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state
        total_rewards += reward

        if done:
            break

    rewards_all_episodes.append(total_rewards)

# Promediar las recompensas cada 100 episodios
rewards_per_hundred_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 100)
average_rewards = [sum(r) / 100 for r in rewards_per_hundred_episodes]

# Mostrar resultado
print("Q-table aprendida:")
print(q_table)

# Graficar
plt.plot(average_rewards)
plt.xlabel("Bloques de 100 episodios")
plt.ylabel("Recompensa promedio")
plt.title("Q-learning con ε-greedy en Taxi-v3")
plt.grid()
plt.show()

