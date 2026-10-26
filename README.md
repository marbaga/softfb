# Codebase for Soft Forward Backward Representations for Zero-shot General Utility Reinforcement Learning (NeurIPS 2026)

## Installation

```
uv sync
```

## Data generation

Datasets will be saved to `./datasets`:
```
# download RND datasets
bash scripts/download.sh walker rnd
bash scripts/download.sh cheetah rnd
bash scripts/download.sh quadruped rnd
bash scripts/download.sh point_mass_maze rnd

# (optionally) replay for consistency with mujoco version
uv run -m scripts.update_data --num-workers 8 --expl_agent rnd --env_name walker
uv run -m scripts.update_data --num-workers 8 --expl_agent rnd --env_name cheetah
uv run -m scripts.update_data --num-workers 8 --expl_agent rnd --env_name quadruped
uv run -m scripts.update_data --num-workers 8 --expl_agent rnd --env_name point_mass_maze

# generate data for bandit experiment
uv run -m scripts.collect_bandit
```

## Running SoftFB on quadruped

```
uv run main.py default.yaml
```

Note: hyperparameters may be set in default.yaml or passed as flags.

Note: in order to run IL evaluations, collect demos with `scripts/collect_demos.py`. This requires a trained FB policy.

Note: training an explicit occupancy model is currently disabled.
