# Connect 4 with AlphaZero-style self-play (STOR 565 final project)

A PyTorch Connect 4 agent built on the open-source
[alpha-zero-general](https://github.com/suragnair/alpha-zero-general) framework (upstream commit
`f1a78e0505c614205f8c60eaa82cd644a07a84b3`, MIT licence). Upstream already contains the Connect 4 rules and a
Keras network; it has **no PyTorch Connect 4 network**, which is what this project adds, together with logging
and an external evaluation.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Training entry point (all hyper-parameters are command-line flags; defaults = the report's settings) |
| `Coach.py`, `MCTS.py`, `Arena.py` | Self-play, PUCT search, arena. Upstream code. We added per-iteration logging (`Coach.py`) and per-game move recording (`Arena.py`); diff against upstream commit `f1a78e05` of AlphaZero-General to see every change. |
| `connect4/Connect4Logic.py`, `Connect4Game.py` | Game rules (upstream, two NumPy-compatibility fixes) |
| `connect4/pytorch/` | **New:** CNN policy/value network and training wrapper (adapted from the Othello PyTorch net) |
| `connect4/test_connect4*.py` | Unit tests (`pytest connect4`) |
| `eval/fixed_opponents.py` | Network vs random / one-step-lookahead / minimax opponents |
| `eval/value_head_uci.py` | Value head vs decision tree on the UCI Connect-4 positions |
| `baseline/decision_tree_baseline.py` | Class baseline: CART on the UCI positions |
| `analysis/make_figures.py` | Arena / loss figures and the numbers quoted in the report |
| `pit.py` | Play against a trained network in the terminal |
| `data/connect-4.data` | UCI Connect-4 data (Tromp, 1995) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > results/requirements-lock.txt      # record exact versions used for the report
```

## Reproduce the report

```bash
# 0. tests (game logic, win detection, UCI encoding)
pytest connect4

# 1. class baseline: decision tree on the UCI data (also writes the train/test split used below)
python baseline/decision_tree_baseline.py data/connect-4.data results

# 2. train (30 iterations x 50 games x 100 simulations, seed 565); logs/ and checkpoints/ go under runs/main
python main.py --run_dir runs/main

# 3. figures and arena statistics
python analysis/make_figures.py --run_dir runs/main --out_dir results

# 4. absolute strength vs fixed opponents (iteration 0 = untrained, an early, a middle and the final network)
python eval/fixed_opponents.py --checkpoints runs/main/checkpoints/checkpoint_0.pth.tar \
    runs/main/checkpoints/checkpoint_<early>.pth.tar runs/main/checkpoints/checkpoint_<mid>.pth.tar \
    runs/main/checkpoints/best.pth.tar --games 100 --procs 4 --out results/fixed_opponents.json

# 5. value head vs the decision tree on the same UCI test split
python eval/value_head_uci.py --data data/connect-4.data --split_dir results \
    --checkpoints runs/main/checkpoints/checkpoint_0.pth.tar runs/main/checkpoints/best.pth.tar \
    --out results/value_head_uci.json
```

Use `ls runs/main/checkpoints` and `runs/main/logs/arena.csv` to see which iterations were accepted
(`checkpoint_<k>.pth.tar` exists only for accepted iterations; `best.pth.tar` is the latest accepted network).

Training log files: `runs/main/logs/arena.csv` (one row per iteration), `losses.csv` (one row per epoch),
`selfplay.csv` (game lengths, example counts, outcomes) and `config.txt` (every setting, plus torch/numpy versions).

## Settings used

| | |
|---|---|
| Iterations / self-play games per iteration | 30 / 50 |
| MCTS simulations per move; `c_puct` | 100; 1.0 (no Dirichlet noise) |
| Temperature | 1 for the first 14 moves of a game, then 0 (`tempThreshold = 15`) |
| Training examples | every move yields the position and its left-right mirror image (2 examples per move); history = last 20 iterations, queue cap 200,000 |
| Epochs per iteration; batch size | 10; 64 (batches sampled with replacement, `len(examples)/64` batches per epoch) |
| Optimiser | Adam, learning rate 0.001, no weight decay; dropout 0.3 |
| Network | 4 conv layers (3x3, 512 filters, batch norm, ReLU) -> FC 1024 -> FC 512 -> policy head (7, softmax) + value head (1, tanh) |
| Loss | `(z - v)^2 - pi . log p` (MSE + cross-entropy, summed, averaged over the batch) |
| Arena | 40 games (20 per colour), temperature 0, promotion if wins / (wins + losses) >= 0.6 (draws ignored) |
| Seed | 565 (Python, NumPy, PyTorch). Results are not bit-for-bit reproducible across machines or PyTorch versions |

## Things to know when reading the results

* **Arena games are not fully independent.** As in upstream, each arena player keeps one search tree for all 40
  games and plays deterministically (temperature 0, random tie-breaks), so some games can repeat.
  `arena.csv` logs `distinct_games` (number of distinct move sequences out of 40) for every iteration.
* **Draws are ignored by the promotion rule** (upstream behaviour). The draw value used in training is 1e-4.
* **Training examples per iteration = 2 x (games x average game length)** because of the mirror augmentation.
* `checkpoint_0.pth.tar` is the randomly initialised network (before any training), used as a reference point.

## Differences from upstream (summary)

1. New PyTorch network and wrapper for Connect 4 (`connect4/pytorch/`).
2. `Coach.py`: writes `arena.csv`, `selfplay.csv`; saves the untrained network as `checkpoint_0`.
3. `Arena.py`: records each game's move sequence (to count distinct games).
4. `Connect4Logic.py` / `Connect4Game.py` / `Connect4Players.py`: `np.int` -> `int`, `tostring` -> `tobytes`
   (removed in recent NumPy), and a `game` -> `self.game` bug fix.
5. `connect4/test_connect4.py`: two stale expectations fixed; `test_connect4_extra.py` added.
6. `main.py` rewritten with command-line flags, seeding and a saved `config.txt`.

## Licence

MIT, inherited from alpha-zero-general (see `LICENSE`).
