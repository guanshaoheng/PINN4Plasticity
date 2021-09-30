import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os

# check if GPU is available and use it; otherwise use CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %% Network initializing
# N is a Neural Network - This is exactly the network used by Lagaris et al. 1997
N = nn.Sequential(nn.Linear(1, 50),
                  nn.Sigmoid(),
                  nn.Linear(50, 1, bias=False))

# Initial condition
A = 0.

# The Psi_t function
Psi_t = lambda x: A + x * N(x)

# The right hand side function
f = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0


# The loss function
def loss(x):
    x.requires_grad = True
    outputs = Psi_t(x)
    Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                                  create_graph=True)[0]
    return torch.mean((Psi_t_x - f(x, outputs)) ** 2)


# %% test
x_np = np.linspace(0, 2, 100, dtype=np.float32)[:, None]
x = torch.from_numpy(x_np).to(device)
x.requires_grad = True
outputs = Psi_t(x)
psi_t_x = torch.autograd.grad(outputs=outputs, inputs=x, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
psi_t_x_np = psi_t_x.detach().numpy()
print(psi_t_x_np.shape)

# %% Optimize (same algorithm as in Lagaris)
optimizer = torch.optim.LBFGS(N.parameters())
# The collocation points used by Lagaris
x = torch.Tensor(np.linspace(0, 2, 100)[:, None])


# Run the optimizer
def closure():
    """
        NOTE: closure fucntion is necessary for the optimizer LBFGS,
              while not for the classical Adam
    """
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l


for i in range(10):
    optimizer.step(closure=closure)
    print('\t Epoch: %d \t loss: %.3e' % (i + 1, loss(x).detach().numpy()))

# Let's compare the result to the true solution
x = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    yy = Psi_t(torch.Tensor(x)).numpy()
yt = np.exp(-x / 5.0) * np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, yt, label='True')
ax.plot(x, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$\Psi(x)$')
plt.legend(loc='best')
plt.show()

# %% Optimizing the network by classical Adam
"""
    Based on my experience, the Adam is much faster than the LBFGS while 
    performs poorer in the differential-related loss value optimizing.
    
    In this problem, only 10 epoches are needed for LBFGS while 1000 for Adam.
"""
N = nn.Sequential(nn.Linear(in_features=1, out_features=50, bias=True),
                  nn.Sigmoid(),
                  nn.Linear(in_features=50, out_features=1, bias=False))
optimizerAdam = torch.optim.Adam(N.parameters())

x = torch.Tensor(np.linspace(0., 2, 100)[:, None])
for i in range(1000):
    optimizerAdam.zero_grad()
    l = loss(x)
    l.backward()
    optimizerAdam.step()
    print('\t Epoch: %d \t loss: %.3e' % (i+1, l.detach().numpy()))

# Let's compare the result to the true solution
x = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    yy = Psi_t(torch.Tensor(x)).numpy()
yt = np.exp(-x / 5.0) * np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, yt, label='True')
ax.plot(x, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$\Psi(x)$')
plt.legend(loc='best')
plt.show()

