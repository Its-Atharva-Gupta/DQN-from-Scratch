# DQN Atari — Breakout

A from-scratch implementation of the **Deep Q-Network (DQN)** algorithm for playing **Atari Breakout**, built with [PyTorch](https://pytorch.org/) and [Gymnasium](https://gymnasium.farama.org/).

This project follows the architecture described in the original DeepMind paper [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013).

---

## Features

- **CNN-based Q-Network** — DeepMind architecture: 3 convolutional layers + 2 fully connected layers
- **Experience Replay** — Stores up to 50k transitions as `uint8` to reduce RAM usage
- **Target Network** — Periodically synced every 1,000 training steps to stabilise learning
- **Epsilon-Greedy Exploration** — Linear decay from 1.0 → 0.05 over 200k steps
- **Reward Clipping** — All rewards clipped to `[-1, 1]`
- **Gradient Clipping** — Gradients clipped at norm 10 to prevent divergence
- **Checkpoint / Resume** — Training state is saved every 50k steps; resume seamlessly from a `.pt` file
- **Logging** — Prints mean episode reward (last 100 episodes), average loss, epsilon, and buffer size every 10k steps

---

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.12.1
- Gymnasium ≥ 1.3.0
- ALE-Py (Arcade Learning Environment) ≥ 0.12.0

All dependencies are listed in `pyproject.toml`. Install them with:

```bash
pip install .
# or, using uv:
uv sync
```

---

## Usage

Train the DQN agent from scratch:

```bash
python main.py
```

To **resume training** from a checkpoint, simply re-run the same command — the script will automatically load `breakout_dqn.pt` if it exists.

### Checkpoints

| File | Description |
|---|---|
| `breakout_dqn.pt` | Full training checkpoint (model weights, optimizer state, step counter, episode rewards). Saved every 50k steps. |
| `breakout_dqn_final.pt` | Final model weights only. Saved at the end of training (1M steps). |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `BUFFER_SIZE` | 50,000 | Replay buffer capacity |
| `BATCH_SIZE` | 32 | Mini-batch size for SGD |
| `GAMMA` | 0.99 | Discount factor |
| `LR` | 1×10⁻⁴ | Adam learning rate |
| `EPSILON_START` | 1.0 | Starting exploration rate |
| `EPSILON_END` | 0.05 | Final exploration rate |
| `EPSILON_DECAY_STEPS` | 200,000 | Steps over which epsilon decays linearly |
| `WARMUP_STEPS` | 10,000 | Steps of random exploration before training starts |
| `TARGET_UPDATE_FREQ` | 1,000 | Steps between target network syncs |
| `TRAIN_FREQ` | 4 | Train the network every N steps |
| `TOTAL_STEPS` | 1,000,000 | Total environment steps |

---

## Network Architecture

The `CNNPolicy` network mirrors the DeepMind DQN design:

| Layer | Output Shape |
|---|---|
| Conv2D (8×8, stride 4), 32 filters → ReLU | 20 × 20 × 32 |
| Conv2D (4×4, stride 2), 64 filters → ReLU | 9 × 9 × 64 |
| Conv2D (3×3, stride 1), 64 filters → ReLU | 7 × 7 × 64 |
| Flatten | 3136 |
| Linear → ReLU | 512 |
| Linear | `n_actions` (4 for Breakout) |

The network takes as input a stack of 4 grayscale 84×84 frames (the last 4 observations).

---

## Project Structure

```
DQN_ATARI/
├── main.py                  # Full DQN training script
├── pyproject.toml           # Python project metadata & dependencies
├── .python-version          # Python version (3.11)
├── .gitignore
├── README.md
├── breakout_dqn.pt          # Training checkpoint (generated)
├── breakout_dqn_final.pt    # Final model weights (generated)
└── .venv/                   # Virtual environment
```

---

## Results

After ~1M steps of training, the agent typically learns to consistently hit the ball and score points by breaking bricks. The mean reward over the last 100 episodes provides a good indicator of policy quality during training.

Example training log output:

```
Step    50,000 | Ep   250 | Mean reward (100ep):    3.25 | Loss: 0.00953 | Epsilon: 0.809 | Buffer: 50,000
Step   100,000 | Ep   520 | Mean reward (100ep):    7.80 | Loss: 0.00721 | Epsilon: 0.594 | Buffer: 50,000
Step   500,000 | Ep  2600 | Mean reward (100ep):   12.40 | Loss: 0.00487 | Epsilon: 0.050 | Buffer: 50,000
```

---

## Acknowledgements

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al.)
- [Gymnasium](https://gymnasium.farama.org/) and [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [PyTorch](https://pytorch.org/)

---

## License

This project is provided for educational and research purposes.
