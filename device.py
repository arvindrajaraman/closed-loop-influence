import torch

device = 'cpu'
if torch.cuda.is_available():
    print('Found GPU')
    device = 'cuda'
else:
    print('Did not find GPU')
