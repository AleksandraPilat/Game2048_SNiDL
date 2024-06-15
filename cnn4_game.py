import torch

import game

from cnn4 import CNN4

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN4().to(device)
    model.load_state_dict(
        torch.load('cnn4_models/cnn4_epoch_59_step_10000000.pt'))

    game.run_game_simulation(model, 'cnn4', device)
