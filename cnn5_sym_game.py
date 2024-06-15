import torch
from pathlib import Path
import game

from cnn5_sym import CNN5Sym

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN5Sym().to(device)
    model.load_state_dict(
        torch.load(f'{Path(__file__).parent}/cnn5_sym_models/cnn5_sym_epoch_59_step_10000000.pt'))

    game.run_game_simulation(model, 'cnn5_sym', device)
