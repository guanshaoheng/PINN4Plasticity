import torch
import torch.nn as nn
from time import perf_counter
from functools import partial
import numpy as np
import os
import matplotlib.pyplot as plt


def grad(outputs, inputs):
    """
    This is useful for taking derivatives
    """
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]


def u_r(x, delta, network1, network2, scale_network_out):
    """
    This function returns displacement field u_r.
    u_r comprises the horizontal and the vertical components
    of displacement (u_x and u_y respectively.)
    -> x should contain the X and Y coordinates. Should be of shape: (batch size x 2)
    -> delta is the Direchlet boundary condition. Should be something similar to: torch.tensor(([right_edge_displacement, 0.]))
    """
    u_x = delta[0] - delta[0] * (1. - x[:, 0][:, None]) + x[:, 0][:, None] * (1. - x[:, 0][:, None]) * network1(
        x) / scale_network_out
    u_y = x[:, 0][:, None] * (1. - x[:, 0][:, None]) * network2(x) / scale_network_out
    u_r = torch.cat((u_x, u_y), dim=1)
    return u_r


def loss(X, delta, network1, network2, scale_network_out, device):
    """
    This is our loss function that we want to minimize.
    -> X: inputs data. Should be of shape: (batch_size x 2)
    -> Displacement function (which requires only X as input).
    """
    X = X.to(device)
    X.requires_grad = True
    uu = u_r(x=X, network1=network1, network2=network2, scale_network_out=scale_network_out, delta=delta)
    g_u_x = grad(uu[:, 0], X)[:, None, :]
    g_u_y = grad(uu[:, 1], X)[:, None, :]
    J = torch.cat([g_u_x, g_u_y], 1)
    I = torch.eye(2).repeat(X.shape[0], 1, 1).to(device)
    F = I + J
    log_det_F = torch.log(F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0])
    res = (0.5 * (torch.einsum('ijk,ijk->i', F, F) - 2) - log_det_F + 50 * log_det_F ** 2)
    return torch.mean(res)


def train_MultiFieldFracture_seperate_net(net1,
                                          net2,
                                          right_edge_displacement,
                                          batch_size_X, max_iter,
                                          print_results_every,
                                          scale_network_out,
                                          device):
    """
    Trains given networks.
    -> right_edge_displacement: corresponds to boundary condition (delta) on the right edge. Example: 0.5
    -> batch_size_X: how many X's do you want to use for each iteration. Example: 10
    -> print_results_every: how often do you want to display the results. Example: 100
    """
    print('\n\nStarting training loop with seperate networks for u_x and u_y...\n\n')
    print('Right Edge Displacement: %.3f' % right_edge_displacement,
          '\t\tbatch_size_X: %d' % batch_size_X,
          '\t\tmax_iter: %d\n' % max_iter)
    network1 = net1
    network1 = network1.to(device)
    network2 = net2
    network2 = network2.to(device)

    # Direchlet boundary condition
    delta = torch.tensor(([right_edge_displacement, 0.]),
                         device=device)  # this contains the x and the y coordinate of the displacement applied on the body
    # Here are the parameters that we would like to optimze
    parameters = list(network1.parameters()) + list(network2.parameters())
    # Initialize the optimizer - Notice that it needs to know about the
    # parameters it it optimizing
    optimizer = torch.optim.Adam(parameters, lr=1e-4)  # lr is the learning rate
    # Records time the loop starts
    start_time = perf_counter()
    # Some place to hold the training loss for visualizing it later
    loss_list = []
    elapsed_time = 0.0
    running_loss = 0.0
    for i in range(max_iter):
        X = torch.distributions.Uniform(0, 1).sample((batch_size_X, 2))
        X = X.to(device)
        # This is essential for the optimizer to keep
        # track of the gradients correctly
        # It is using some buffers internally that need to
        # be manually zeroed on each iteration.
        # This is because the optimizer doesn't know when you are done with the
        # calculation of the loss.
        optimizer.zero_grad()
        # Evaluate the loss - That's what you are minimizing
        l = loss(X=X,
                 delta=delta,
                 network1=network1, network2=network2,
                 scale_network_out=scale_network_out,
                 device=device)
        # Add the loss to the running loss
        running_loss += l.item()
        # Evaluate the derivative of the loss with respect to
        # all parameters
        l.backward()
        # And now you are ready to make a step
        optimizer.step()
        if (i + 1) % print_results_every == 0:
            # Print loss, time elapsed every "print_results_every"# iterations
            current_time = perf_counter()
            elapsed_time = current_time - start_time
            print('[iter: %d]' % (i + 1), '\t\telapsed_time: %3d secs' % elapsed_time, '\t\tLoss: ',
                  running_loss / print_results_every)
            loss_list.append(running_loss / print_results_every)
            running_loss = 0.0

    return loss_list, network1, network2


def model_capacity(net):
    """
    Prints the number of parameters and the number of layers in the network
    -> Requires a neural network as input
    """
    number_of_learnable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_layers = len(list(net.parameters()))
    print("\nThe number of layers in the model: %d" % num_layers)
    print("The number of learnable parameters in the model: %d\n" % number_of_learnable_params)


def plot_loss(loss, label, ax):
    """
    Plots the loss function.
    -> loss: list containing the losses
    -> label: label for this loss
    """
    ax.plot(100 * np.arange(len(loss)), loss, label='%s' % label)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    plt.legend(loc='best')


def plot_displacement(network1, network2,
                      right_edge_displacement,
                      scale_network_out,
                      num_samples_X=200, s=4, figTitle='Init',
                      ):
    """
    Plots the horizontal and vertical components of displacements at given number of points.
    -> network1: neural network 1
    -> network2: neural network 2
    -> right_edge_displacement: corresponds to boundary condition (delta) on the right edge. Example: 0.5
    -> num_samples_X: number of grid points
    -> s: Marker size
    """
    network1 = network1.to(torch.device('cpu'))
    network2 = network2.to(torch.device('cpu'))
    ## Direchlet Boundary condition
    delta = torch.tensor(([right_edge_displacement,
                           0.]))  # this contains the x and the y coordinate of the displacement applied on the body

    for i in range(1):
        figure, ax = plt.subplots(1, 2, figsize=(20, 5))

        X_init = torch.linspace(0, 1, num_samples_X)
        xx, yy = torch.meshgrid(X_init, X_init)
        X_init = torch.cat((xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]), axis=1)

        displacement = u_r(X_init, delta, network1, network2, scale_network_out).detach().numpy()
        X_final_dual = X_init + displacement
        X_final_dual = X_final_dual.numpy()

        sc1 = ax[0].scatter(X_final_dual[:, 0], X_final_dual[:, 1], s=s, c=displacement[:, 0],
                            cmap=plt.cm.get_cmap('copper'))
        plt.colorbar(sc1, ax=ax[0])
        ax[0].set_title('Horiontal Displacement ($u_x$)')

        sc2 = ax[1].scatter(X_final_dual[:, 0], X_final_dual[:, 1], s=s, c=displacement[:, 1],
                            cmap=plt.cm.get_cmap('copper'))
        plt.colorbar(sc2, ax=ax[1])
        ax[1].set_title('Vertical Displacement ($u_y$)')

        plt.tight_layout()
        plt.savefig('%s' % figTitle)
        # plt.show()

