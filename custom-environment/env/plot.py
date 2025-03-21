import pickle
import matplotlib.pyplot as plt

# Pickle-Datei laden
with open("training_data.pkl", "rb") as f:
    Q, TD_error_per_episode, reward_per_episode = pickle.load(f)


plt.figure(figsize=(8, 5))
plt.plot(TD_error_per_episode, label="TD-Error", color="red")
plt.xlabel("Episode")
plt.ylabel("TD-Error")
plt.title("TD-Error pro Episode")
plt.legend()
plt.grid()
plt.show()