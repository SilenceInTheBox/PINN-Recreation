import torch
import matplotlib.pyplot as plt
import numpy as np


def source(x):
    '''Source term in paper. Implemented to work with:
        1. a single 2d vector (x.shape == 1)
        2. batched 2d vectors (x.shape == 2)
        3. --- (x.shape == 3)'''
    f = 0
    if len(x.shape) == 1:
        x_0 = x[0]
        x_1 = x[1]
    elif len(x.shape) == 2:
        x_0 = x[:, 0]
        x_1 = x[:, 1]
    elif len(x.shape) == 3:
        x_0 = x[:, :, 0]
        x_1 = x[:, :, 1]

    for k in range(1, 5):
        f += (-1)**(k+1) * 2*k * torch.sin(k *
                                           np.pi*x_0) * torch.sin(k*np.pi*x_1)

    return f / 4


def get_closure(optimizer, batch_size, model, us, lossi, count):
    '''
    Returns the closure function with necessary parameters for LBFGS optimizer.'''
    def closure():
        optimizer.zero_grad()

        # gradient accumulation
        for _ in range(batch_size):
            # inputs
            x = torch.randn((2,), requires_grad=True)

            # derivates wrt inputs
            hessian = torch.autograd.functional.hessian(
                model, x, create_graph=True, strict=True)
            u_xx = hessian[0, 0]
            u_yy = hessian[1, 1]
            us['u_xx'].append(u_xx.item())
            us['u_yy'].append(u_yy.item())

            # loss fct
            loss = u_xx + u_yy - source(x)
            lossi.append(loss.item())
            count += 1

            loss.backward()
        return loss

    return closure


class PINN(torch.nn.Module):
    '''Model orients itself on draft by paper, 
    however it is fully customizable.'''
    n_hidden = 50

    def __init__(self) -> None:
        super().__init__()

        self.approximator = torch.nn.Sequential(torch.nn.Linear(2, self.n_hidden, bias=True),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(
                                                    self.n_hidden, self.n_hidden, bias=True),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(
                                                    self.n_hidden, self.n_hidden, bias=True),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(self.n_hidden, 1, bias=False))

    def forward(self, x):
        return self.approximator(x)
