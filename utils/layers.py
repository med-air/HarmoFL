import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from torch import nn
import torch
from .amp_utils import process

class AmpNorm(nn.Module):
    def __init__(self, input_shape, momentum=0.1):
        super(AmpNorm, self).__init__()
        self.register_buffer('running_amp', torch.zeros(input_shape))
        self.momentum = momentum
        self.fix_amp = False     
        
    def forward(self, x):
        device = x.device
        if not self.fix_amp:
            if torch.sum(self.running_amp) == 0:
                x, amp = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)
                self.running_amp = torch.from_numpy(amp)
            else:
                x, amp = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)
                self.running_amp = torch.from_numpy(amp)
        else:
            x, _ = process(x.cpu().numpy(), self.running_amp.cpu().numpy(), self.momentum, self.fix_amp)

        return torch.from_numpy(x).to(device)

if __name__ =='__main__':
    exit()

