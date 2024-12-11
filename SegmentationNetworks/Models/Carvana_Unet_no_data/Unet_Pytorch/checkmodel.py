import torch

checkpoint = torch.load('my_checkpoint.pth.tar', map_location=torch.device('cpu'))
print(checkpoint)