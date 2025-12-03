import torch
from torch.nn import Linear

class Lin(torch.nn.Module):
    def __init__(self, args):
        super(Lin, self).__init__()
        self.lin = Linear(args.num_features, args.output, bias=True)
        self.prelu = torch.nn.PReLU(args.output)

    def forward(self, x):
        z = self.lin(x)
        z = self.prelu(z)
        return z
