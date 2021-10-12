import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os
from util.example2_utils import grad, u_r, loss, train_MultiFieldFracture_seperate_net, \
    plot_loss, plot_displacement, model_capacity
from util.denseNet import DenseResNet

# check if GPU is available and use it; otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# %% Initial configuration
# Boundary condition (delta) on the right edge
right_edge_displacement = 0.5  # try 0.05
# Scale down the network output to ensure we get +ve determinant of the Jacobian.
# We have to scale the output so that as the training begins we don't initialize a displacement
# which has no physical meaning.
# For example, the determinant of the Jacobian cannot be negative
# since that would mean negative volume; which has no physical meaning.
scale_network_out = 5

# # %% Network initializing
# # here is the network for the u_x
# simple_net1 = nn.Sequential(nn.Linear(2, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 1)
#                             )
# # here is the network for the u_y
# simple_net2 = nn.Sequential(nn.Linear(2, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 50), nn.Sigmoid(),
#                             nn.Linear(50, 1)
#                             )
#
# # here is how we can find the number of layers and number of model parameters in each network
# model_capacity(simple_net1)
# model_capacity(simple_net2)
#
# # %% Plot initial configuration
# plot_displacement(simple_net1, simple_net2,
#                   right_edge_displacement, scale_network_out, figTitle='initDense')
#
# # %% train the network
# loss_list_simple_net, \
# simple_net1, \
# simple_net2 = train_MultiFieldFracture_seperate_net(net1=simple_net1,
#                                                     net2=simple_net2,
#                                                     right_edge_displacement=right_edge_displacement,
#                                                     batch_size_X=10,
#                                                     max_iter=5000,
#                                                     print_results_every=100,
#                                                     scale_network_out=scale_network_out,
#                                                     device=device)
#
# # %% visualizing the training loss
# figure, ax = plt.subplots(dpi=100)
# plot_loss(loss=loss_list_simple_net, label='Simple Net', ax=ax)
#
# if right_edge_displacement == 0.05:
#     ax.plot(100 * np.arange(len(loss_list_simple_net)), 0.006 * np.ones(len(loss_list_simple_net)), label='FEM Energy')
# elif right_edge_displacement == 0.5:
#     ax.plot(100 * np.arange(len(loss_list_simple_net)), 0.43 * np.ones(len(loss_list_simple_net)), label='FEM Energy')
# plt.legend(loc='best')
# plt.show()
#
# # %% Visualizing the body post-training
# plot_displacement(network1=simple_net1,
#                   network2=simple_net2,
#                   right_edge_displacement=right_edge_displacement,
#                   scale_network_out=scale_network_out, figTitle='simpleNetPostTrain')
#
# # %% Dense net initilizing
# dense_net1 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=5,
#                          num_layers_per_block=3, num_neurons=50, activation=nn.Sigmoid(),
#                          fourier_features=False, m_freqs=100, sigma=10, tune_beta=False, device=device)
#
# dense_net2 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=5,
#                          num_layers_per_block=3, num_neurons=50, activation=nn.Sigmoid(),
#                          fourier_features=False, m_freqs=100, sigma=10, tune_beta=False, device=device)
#
# # Here is the model capacity:
# model_capacity(dense_net1)
# model_capacity(dense_net2)
#
# # %% Let's visualize the body pre-training and make sure the boundary conditions are satified
# plot_displacement(dense_net1, dense_net2, right_edge_displacement, scale_network_out, figTitle='denseInit')
#
# # %% Dense net training
# dense_net1.changeDevice(device)
# dense_net2.changeDevice(device)
# loss_list_dense, dense_net1, \
# dense_net2 = train_MultiFieldFracture_seperate_net(net1=dense_net1,
#                                                    net2=dense_net2,
#                                                    right_edge_displacement=right_edge_displacement,
#                                                    batch_size_X=10,
#                                                    max_iter=5000,
#                                                    print_results_every=100,
#                                                    scale_network_out=scale_network_out,
#                                                    device=device)
# dense_net1.changeDevice(torch.device('cpu'))
# dense_net2.changeDevice(torch.device('cpu'))
#
# # %% Let's visualize the training loss
# figure, ax = plt.subplots(dpi=100)
# plot_loss(loss_list_simple_net, 'Simple Net', ax)
# plot_loss(loss_list_dense, 'ResNet', ax)
#
# if right_edge_displacement == 0.05:
#     ax.plot(100 * np.arange(len(loss_list_dense)), 0.006 * np.ones(len(loss_list_dense)), label='FEM Energy')
# elif right_edge_displacement == 0.5:
#     ax.plot(100 * np.arange(len(loss_list_dense)), 0.43 * np.ones(len(loss_list_dense)), label='FEM Energy')
# plt.legend(loc='best')
#
# # %% Let's visualize the body post-training
# plot_displacement(dense_net1, dense_net2, right_edge_displacement, scale_network_out, figTitle='residualNetPostTrain')

# %% Fourier dense net
Fourier_dense_net1 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=3,
                                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(),
                                 fourier_features=True, m_freqs=100, sigma=10, tune_beta=False)

Fourier_dense_net2 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=3,
                                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(),
                                 fourier_features=True, m_freqs=100, sigma=10, tune_beta=False)

# Here is the model capacity:
model_capacity(Fourier_dense_net1)
model_capacity(Fourier_dense_net2)

# Let's visualize the body pre-training and make sure the boundary conditions are satified
plot_displacement(Fourier_dense_net1,
                  Fourier_dense_net2,
                  right_edge_displacement,
                  scale_network_out,
                  figTitle='fourierInit')
Fourier_dense_net1.changeDevice(device)
Fourier_dense_net2.changeDevice(device)
loss_list_dense_fourier, Fourier_dense_net1, Fourier_dense_net2 = train_MultiFieldFracture_seperate_net(
    net1=Fourier_dense_net1,
    net2=Fourier_dense_net2,
    right_edge_displacement=right_edge_displacement,
    batch_size_X=10,
    max_iter=10000,
    print_results_every=100, scale_network_out=scale_network_out,
    device=device)
Fourier_dense_net1.changeDevice(torch.device('cpu'))
Fourier_dense_net2.changeDevice(torch.device('cpu'))

## Plot the loss
# figure, ax = plt.subplots(dpi=100)
# plot_loss(loss_list_simple_net, 'Simple Net', ax=ax)
# plot_loss(loss_list_dense, 'ResNet', ax=ax)
# plot_loss(loss_list_dense_fourier, 'Fourier-ResNet', ax=ax)
# plt.savefig('./lossComparing.svg')

# if right_edge_displacement==0.05:
#     ax.plot(100*np.arange(len(loss_list_dense)), 0.006*np.ones(len(loss_list_dense)), label='FEM Energy')
# elif right_edge_displacement==0.5:
#     ax.plot(100*np.arange(len(loss_list_dense)), 0.43*np.ones(len(loss_list_dense)), label='FEM Energy')

plt.legend(loc='best')
# Let's visualize the body post-training
plot_displacement(Fourier_dense_net1,
                  Fourier_dense_net2,
                  right_edge_displacement,
                  scale_network_out,
                  figTitle='FourierNetPostTrain')


# %% Simple fourier network
Fourier_simple_net1 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=3,
                                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(),
                                 fourier_features=True, m_freqs=100, sigma=10, tune_beta=False, simpleLinearFlag=True)

Fourier_simple_net2 = DenseResNet(dim_in=2, dim_out=1, num_resnet_blocks=3,
                                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(),
                                 fourier_features=True, m_freqs=100, sigma=10, tune_beta=False, simpleLinearFlag=True)

# Here is the model capacity:
model_capacity(Fourier_simple_net1)
model_capacity(Fourier_simple_net2)

Fourier_simple_net1.changeDevice(device)
Fourier_simple_net2.changeDevice(device)
loss_list_simple_fourier, Fourier_simple_net1, Fourier_simple_net2 = train_MultiFieldFracture_seperate_net(
    net1=Fourier_simple_net1,
    net2=Fourier_simple_net2,
    right_edge_displacement=right_edge_displacement,
    batch_size_X=10,
    max_iter=10000,
    print_results_every=100, scale_network_out=scale_network_out,
    device=device)
Fourier_simple_net1.changeDevice(torch.device('cpu'))
Fourier_simple_net2.changeDevice(torch.device('cpu'))

figure, ax = plt.subplots(dpi=100)
# plot_loss(loss_list_simple_net, 'Simple Net', ax=ax)
# plot_loss(loss_list_dense, 'ResNet', ax=ax)
plot_loss(loss_list_dense_fourier, 'Fourier-ResNet', ax=ax)
plot_loss(loss_list_simple_fourier, 'Fourier-simple', ax=ax)
plt.savefig('./lossComparing.svg')

if right_edge_displacement==0.05:
    ax.plot(100*np.arange(len(loss_list_dense_fourier)), 0.006*np.ones(len(loss_list_dense_fourier)), label='FEM Energy')
elif right_edge_displacement==0.5:
    ax.plot(100*np.arange(len(loss_list_dense_fourier)), 0.43*np.ones(len(loss_list_dense_fourier)), label='FEM Energy')

plt.legend(loc='best')
# Let's visualize the body post-training
plot_displacement(Fourier_simple_net1,
                  Fourier_simple_net2,
                  right_edge_displacement,
                  scale_network_out,
                  figTitle='FourierSimplePostTrain')
