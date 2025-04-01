import pickle
import matplotlib.pyplot as plt

file_paths = {
    "C=1": "training_data_C1.pkl",
    "C=0.1": "training_data_C0.1.pkl",
    "C=0.01": "training_data_C0.01.pkl"
}

colors = ["red", "blue", "green"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

for (label, file), color in zip(file_paths.items(), colors):
    with open(file, "rb") as f:
        Q, TD_error_per_episode, reward_per_episode = pickle.load(f, fix_imports=True, encoding="latin1")

    
    ax1.plot(TD_error_per_episode, color=color, label=label)
    
    
    ax2.plot(reward_per_episode, color=color, label=label)


ax1.set_xlabel("Episodes")
ax1.set_ylabel("TD-Error")
ax1.set_title("Average absolute Bellman error per 100 steps")
ax1.set_ylim([0.0, 4])
ax1.set_xlim([0, 100])
ax1.grid()
ax1.legend()


ax2.set_xlabel("Episodes")
ax2.set_ylabel("Reward")
ax2.set_title("Average reward per 100 steps")
ax2.set_ylim([-0.025, -0.000])
ax2.set_xlim([0, 90])
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()