import torch

import game

from mlp5 import MLP5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP5().to(device)
    model.load_state_dict(
        torch.load('mlp5_models/mlp5_epoch_59_step_10000000.pt'))

    game.run_game_simulation(model, 'mlp4', device)
