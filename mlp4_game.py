import torch
from pathlib import Path
import game

from mlp4 import MLP4

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP4().to(device)
    model.load_state_dict(
        torch.load(f'{Path(__file__).parent}/mlp4_models/mlp4_epoch_59_step_10000000.pt'))

    game.run_game_simulation(model, 'mlp4', device)
