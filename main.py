import numpy as np
import gymnasium as gym
import ale_py
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
from collections import deque
import random
import os

gym.register_envs(ale_py)

# ── Device ────────────────────────────────────────────────────────────────────
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BUFFER_SIZE         = 50_000     # ~1.4GB RAM stored as uint8
BATCH_SIZE          = 32
GAMMA               = 0.99
LR                  = 1e-4
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 200_000   # steps over which epsilon decays
WARMUP_STEPS        = 10_000    # fill buffer before training starts
TARGET_UPDATE_FREQ  = 1_000     # sync target net every N steps
TRAIN_FREQ          = 4         # train every N steps
TOTAL_STEPS         = 1_000_000
SAVE_EVERY          = 50_000
CHECKPOINT_PATH     = "breakout_dqn.pt"
LOG_EVERY           = 10_000

# ── Environment ───────────────────────────────────────────────────────────────
env = gym.make('ALE/Breakout-v5', obs_type="grayscale")
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, 4)

N_ACTIONS = env.action_space.n
print(f"Action space: {N_ACTIONS}")

# ── CNN (DeepMind DQN architecture) ───────────────────────────────────────────
class CNNPolicy(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

online_net = CNNPolicy(N_ACTIONS).to(device)
target_net = CNNPolicy(N_ACTIONS).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()   # never backprop through target net

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# ── Replay Buffer ─────────────────────────────────────────────────────────────
# Stores: (state, action, new_state, reward, done)
# States kept as uint8 numpy to save RAM — converted to float32 at sample time
replay_buffer: deque = deque(maxlen=BUFFER_SIZE)

# ── Load checkpoint if it exists ──────────────────────────────────────────────
start_step = 0
episode_rewards_all = []

if CHECKPOINT_PATH:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    online_net.load_state_dict(ckpt["model_state_dict"])
    target_net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step          = ckpt["step"]
    episode_rewards_all = ckpt.get("episode_rewards", [])
    print(f"Resumed from step {start_step:,}")
else:
    print("No checkpoint found — training from scratch.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_epsilon(step: int) -> float:
    if step < WARMUP_STEPS:
        return EPSILON_START
    progress = min(1.0, (step - WARMUP_STEPS) / EPSILON_DECAY_STEPS)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)

def sample_batch():
    batch                                        = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, new_states, rewards, dones  = zip(*batch)

    # Stack numpy arrays then convert — much faster than looping torch.cat
    states_t     = torch.from_numpy(np.stack(states)).float().to(device)     / 255.0
    new_states_t = torch.from_numpy(np.stack(new_states)).float().to(device) / 255.0
    actions_t    = torch.tensor(actions, dtype=torch.long).to(device)         # (B,)
    rewards_t    = torch.tensor(rewards, dtype=torch.float32).to(device)      # (B,)
    dones_t      = torch.tensor(dones,   dtype=torch.float32).to(device)      # (B,)

    return states_t, actions_t, new_states_t, rewards_t, dones_t

def train_step() -> float:
    """One Bellman update using a sampled batch."""
    states, actions, new_states, rewards, dones = sample_batch()

    # Q(s, a) for the action actually taken
    q_values = online_net(states)                                    # (B, N_ACTIONS)
    q_sa     = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # Target: r + gamma * max_a' Q_target(s', a')  — masked to 0 if done
    with torch.no_grad():
        next_q  = target_net(new_states).max(dim=1).values          # (B,)
        targets = rewards + GAMMA * next_q * (1.0 - dones)          # (B,)

    # Huber loss — more robust to outlier rewards than MSE
    loss = F.huber_loss(q_sa, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)  # prevent exploding grads
    optimizer.step()

    return loss.item()

# ── Training Loop ─────────────────────────────────────────────────────────────
state, info    = env.reset()
episode_reward = 0.0
episode        = 0
losses         = []

print(f"\nWarmup: {WARMUP_STEPS:,} steps | Total: {TOTAL_STEPS:,} steps\n")

for step in range(start_step + 1, TOTAL_STEPS + 1):
    epsilon = get_epsilon(step)

    # ── Epsilon-greedy action ──────────────────────────────────────────────────
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
            action  = online_net(state_t).argmax().item()

    # ── Step environment ───────────────────────────────────────────────────────
    new_state, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward

    # Clip reward to [-1, 1] — standard for Atari DQN, stabilises training
    clipped_reward = float(np.clip(reward, -1.0, 1.0))
    done           = terminated or truncated

    # ── Store transition ───────────────────────────────────────────────────────
    replay_buffer.append((state, action, new_state, clipped_reward, float(done)))
    state = new_state

    # ── Train ──────────────────────────────────────────────────────────────────
    if step >= WARMUP_STEPS and step % TRAIN_FREQ == 0 and len(replay_buffer) >= BATCH_SIZE:
        loss = train_step()
        losses.append(loss)

    # ── Sync target network ────────────────────────────────────────────────────
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # ── Episode end ────────────────────────────────────────────────────────────
    if done:
        episode += 1
        episode_rewards_all.append(episode_reward)
        episode_reward = 0.0
        state, info    = env.reset()

    # ── Logging ────────────────────────────────────────────────────────────────
    if step % LOG_EVERY == 0:
        mean_r = np.mean(episode_rewards_all[-100:]) if episode_rewards_all else 0.0
        mean_l = np.mean(losses[-500:])              if losses              else 0.0
        print(
            f"Step {step:>8,} | "
            f"Ep {episode:>5} | "
            f"Mean reward (100ep): {mean_r:>7.2f} | "
            f"Loss: {mean_l:.5f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(replay_buffer):>6,}"
        )

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    if step % SAVE_EVERY == 0:
        torch.save({
            "step":                 step,
            "model_state_dict":     online_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode_rewards":      episode_rewards_all,
        }, CHECKPOINT_PATH)
        print(f"  └─ Checkpoint saved at step {step:,}")

env.close()
torch.save(online_net.state_dict(), "breakout_dqn_final.pt")
print("Final model saved → breakout_dqn_final.pt")


