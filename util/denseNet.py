import torch
import torch.nn as nn


class DenseResNet(nn.Module):
    """
    This is a ResNet Class.
    -> dim_in: network's input dimension
    -> dim_out: network's output dimension
    -> num_resnet_blocks: number of ResNet blocks
    -> num_layers_per_block: number of layers per ResNet block
    -> num_neurons: number of neurons in each layer
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> fourier_features: whether to pass the inputs through Fourier mapping. E.g. True or False
    -> m_freq: how many frequencies do you want the inputs to be mapped to
    -> sigma: controls the spectrum of frequencies.
              If sigma is greater more frequencies are consider.
              You can also look at it as sampling from the standard normal, Z~N(0, 1),
              and mapping to another normal, X~N(\mu, \sigma^2), using x = mu + sigma*z.
    -> tune_beta: do you want to consider the parameter beta in the activation functions in each layer? E.g., Tanh(beta*x).
                  In practice it is observed that training beta (i.e. tune_beta=True) could improve convergence.
                  If tune_beta=False, you get the a fixed beta i.e. beta=1.
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3,
                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(),
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False, device=torch.device('cpu'),
                 simpleLinearFlag=False):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.device = device
        self.simpleLinearFlag = simpleLinearFlag

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))

        else:
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features=num_neurons, out_features=num_neurons)
                for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)])

        self.last = nn.Linear(num_neurons, dim_out)

        if fourier_features:
            self.B = nn.Parameter(sigma*torch.randn(dim_in, m_freqs)) # to converts inputs to m_freqs
            self.first = nn.Linear(2*m_freqs, num_neurons)

    def forward(self, x):
        if self.fourier_features:
            cosx = torch.cos(torch.matmul(x, self.B))
            sinx = torch.sin(torch.matmul(x, self.B))
            x = torch.cat((cosx, sinx), dim=1)
            x = self.activation(self.beta0*self.first(x))

        else:
            x = self.activation(self.beta0*self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0]*self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j]*self.resblocks[i][j](z))

            if self.simpleLinearFlag:
                x = z
            else:
                x = z + x

        out = self.last(x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print("\n\nThe number of layers in the model: %d" % num_layers)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    def changeDevice(self, device):
        self.beta0 = self.beta0.to(device)
        self.beta = self.beta.to(device)
