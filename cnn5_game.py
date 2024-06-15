import torch

import game

from cnn5 import CNN5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN5().to(device)
    model.load_state_dict(
        torch.load('cnn5_models/cnn5_epoch_59_step_10000000.pt'))

    game.run_game_simulation(model, 'cnn5', device)
