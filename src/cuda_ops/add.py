import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

# import CUDA ops via JIT
module_path = os.path.dirname(__file__)
add = load(
    name="add",
    sources=[
        os.path.join(module_path, "add_op.cpp"), 
        os.path.join(module_path, 'kernels', "add_kernel.cu")
    ],
)


class AddModelFunction(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, n: int):
        c = torch.empty(n, device='cuda:0')
        
        add.torch_launch_add(c, a, b, n)
        
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None


class AddModel(nn.Module):
    
    def __init__(self, n):
        super(AddModel, self).__init__()

        self.n = n
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)
        
    def forward(self):
        a2 = torch.square(self.a)
        b2 = torch.square(self.b)

        c = AddModelFunction.apply(a2, b2, self.n)
        
        return c


if __name__ == '__main__':
    
    from rich import print
    
    n = 1024
    
    print('Initializing model...')
    model = AddModel(n)
    model.to(device='cuda:0')
    
    
    print('Initializing optimizer...')
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print('Begin training...')
    for epoch in range(500):
        opt.zero_grad()
        y = model()
        loss = y.sum()
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            print(f'Epoch {epoch}: loss = {loss.item()}')

    print('[b green]Training complete.')