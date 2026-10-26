from collections import defaultdict
import numpy as np
from pathlib import Path
from tqdm import trange
import tyro

from custom_dmc.bandit import Bandit

NUM_TRAIN_EPISODES = 10000
HORIZON = 2


def main(replay_buffer_dir: str = "datasets"):
    env = Bandit()
    np.random.seed(0)
    for ep_idx in trange(NUM_TRAIN_EPISODES):
        dataset = defaultdict(list)
        ob, _ = env.reset()
        action = np.zeros((2,))
        for _ in range(HORIZON+1):
            dataset['observation'].append(ob)
            dataset['action'].append(action)
            dataset['reward'].append(np.zeros((1,)))
            dataset['discount'].append(np.ones((1,)))
            dataset['physics'].append(ob)
            action = np.random.uniform(size=(2,)) * 2 - 1
            ob, _, _, _, _ = env.step(action)
        dataset = {k: np.stack(v, 0) for k, v in dataset.items()}
        datasets_dir = Path.cwd() / replay_buffer_dir / "bandit" / "rnd" / "buffer_updated"
        np.savez_compressed(datasets_dir / f'episode_{ep_idx}_{HORIZON}.npz', **dataset)


if __name__ == "__main__":
    tyro.cli(main)
