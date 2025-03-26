import pickle
import matplotlib.pyplot as plt

# Pickle-Datei laden
with open("training_data_c_0.01.pkl", "rb") as f:
    Q, TD_error_per_episode, reward_per_episode = pickle.load(f, fix_imports=True, encoding="latin1")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Erster Plot: TD-Fehler
ax1.plot(TD_error_per_episode, color="red")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("TD-Error")
ax1.set_title('Average absolute Bellman error per 100 steps')
ax1.set_ylim([0.5,8])
ax1.set_xlim([0,100])
ax1.grid()

# Zweiter Plot: Belohnung
ax2.plot(reward_per_episode)
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Reward")
ax2.set_title('Average reward per 100 steps')
ax2.set_ylim([-0.025,-0.005])
ax2
ax2.set_xlim([0,100])
ax2.grid()

# Layout anpassen und anzeigen
plt.tight_layout()
plt.show()