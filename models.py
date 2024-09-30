import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import itertools
import torch.nn.functional as F


"""
Diffusion
"""

class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim=10, max_length=1000, device='cuda'):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self._max_period = 10000.0
        # Compute the positional encodings once in log space
        self.positional_encodings = self._get_positional_encodings().to(device)
        # self.time_embedding()

    def _get_positional_encodings(self):
        pe = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-np.log(self._max_period) / self.embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.reshape(-1, self.embedding_dim)
        return pe
    
    def get_timestep_embedding(self, timesteps):
        """
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
    
        if timesteps.shape[0] != 1:
            timesteps = timesteps.squeeze()
        assert len(timesteps.shape) == 1

        half_dim = self.embedding_dim // 2
        emb = np.log(self._max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = torch.pad(emb, [0, 1], value=0.0)
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb

    
    def time_embedding(self): # is equal to _get_positional_encodings
        

        # Trigonometric encoding following [1].
        pos = torch.arange(self.max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.embedding_dim, 2) * -np.log(self._max_period) / self.embedding_dim)
        pe = torch.zeros(self.max_length, self.embedding_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, time):
        # Add positional embeddings to the input tensor

        return self.positional_encodings[time.squeeze().long(), :]


class ToyNetwork1(nn.Module):   #best_result - Diffusion Visualization
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a embedding encoding time.

        Do not Apply FC net on time positional embedding
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyNetwork1, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps, 
            device=device
        )
        # Make the network
        self.network = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        # Map the time through a positional encoding
        time_embedding = self.positional_embedding(time)
        return self.network(
            torch.cat([x, time_embedding], dim=-1)
        )


def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')

class ToyNetwork2(nn.Module):   # best result - diffusion model hands on
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a embedding encoding time.

        Apply FC net on time positional embedding
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyNetwork2, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Sequential(
            nn.Linear(data_dim, hidden_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, data_dim)
        )
        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        # Map the time through a positional encoding
        time_embedding = self.positional_embedding(time)
        # return self.network(
        #     torch.cat([x, time_embedding], dim=-1)
        # )
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        x_ = self.net1(x)
        # t = process_single_t(x, t)
        time_embedding = self.positional_embedding(time)
        t_emb = self.time_emb_net(time_embedding)
        x = torch.cat([x_, t_emb], dim=-1)
        return self.net2(x)

class ToyNetwork3(nn.Module):   # good result - diffusion on a toy spiral dataset [classifier-free-guide]
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a embedding encoding time.

        Do not apply FC net on time positional embedding
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyNetwork3, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        self.total_timesteps = total_timesteps
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Sequential(
            nn.Linear(data_dim + time_dim, 2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*hidden_dim, data_dim)
        )
        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        # Map the time through a positional encoding
        time_embedding = self.positional_embedding(time)
        x = self.net1(torch.cat([x, time_embedding], dim=-1))
        # x = self.net1(torch.cat([x, time/self.total_timesteps], dim=-1)) # without positional encoding
        return x

class DiffusionBlock(nn.Module):

    def __init__(self, hidden_dim):
        super(DiffusionBlock, self).__init__()
       
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True))
        
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        # x = nn.functional.relu(x)
        return x

class ToyNetwork4(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyNetwork4, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyNetwork5(nn.Module):   # best result - Simple_diffusion_two_moons
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a one-hot time embedding.
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyNetwork5, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        self.total_timesteps = total_timesteps
        # Make the positional embedding
        self.one_hot_encoding = nn.Linear(total_timesteps, 1)
        # Make the network
       
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),   # Input: Noisy data x_t and t
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.LeakyReLU(inplace=True),
            # Output: Predicted noise that was added to the original data point
            nn.Linear(hidden_dim//4, data_dim)
        )

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        # Encode the time index t as one-hot and then use one layer to encode
        # into a single value
        t_embedding = self.one_hot_encoding(
            nn.functional.one_hot(time.long(), num_classes=self.total_timesteps).to(torch.float))
        inp = torch.cat([x, t_embedding], dim=1)
        out = self.net(inp)

        return out

class ToyNetwork6(nn.Module):          # Deep_Unsupervised_Learning_using_Nonequilibrium_Thermodynamics
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(ToyNetwork6, self).__init__()

        self.total_timesteps = total_timesteps
        self.net1 = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.net_t_list = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                         nn.ReLU(), nn.Linear(hidden_dim, data_dim)
                                                         ) for _ in range(total_timesteps)])

    def forward(self, x, time):
        h = self.net1(x)
        return self.net_t_list[time.long()](h)



class ToyDDPM(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyDDPM, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out
   
class ToyScoreNet(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        For predicting score of distribution (with low level sigma)
    """
    def __init__(self, data_dim=2, hidden_dim=128, num_hidden=4, device='cuda'):
        super(ToyScoreNet, self).__init__()
        self.data_dim = data_dim
        

        # Make the network
        self.net1 = nn.Linear(data_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)


    def forward(self, x):
        """
            Forward pass of the network
        """
        x_ = self.net1(x) 
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyScoreNetWithTime(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        For predicting score of distribution (with multi level sigma)
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyScoreNetWithTime, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyBoostingCmpsSep(nn.Module):          # 
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(ToyBoostingCmps, self).__init__()

        self.total_timesteps = total_timesteps
        self.net1 = nn.Sequential(nn.Linear(data_dim * 2, hidden_dim), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU())
        
        self.net_t_list = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        # nn.Linear(hidden_dim, hidden_dim),
                                                        # nn.ReLU(), 
                                                        nn.Linear(hidden_dim, data_dim)
                                                         ) for _ in range(total_timesteps)])

    def forward(self, z, x, time):
        h = self.net1(torch.concat([z, x], dim=-1))
        return self.net_t_list[time](h)

class ToyBoostingCmps(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyDDPM, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out
  
class ToyBoosting(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyBoosting, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time.long()])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyBoostingTimeEmb(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (with time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyBoostingTimeEmb, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        time_embedding = self.positional_embedding(time.long())
        x_ = self.net1(torch.hstack([x, time_embedding])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyBoostingSepShrd(nn.Module):          # 
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(ToyBoostingSepShrd, self).__init__()

        self.total_timesteps = total_timesteps
        self.net1 = nn.Sequential(nn.Linear(data_dim, hidden_dim), 
                                    nn.ReLU(),
                                    # nn.Linear(hidden_dim, hidden_dim), 
                                    # nn.ReLU(),
                                    # nn.Linear(hidden_dim, hidden_dim),
                                    # nn.ReLU()
                                    )
        
        self.net_t_list = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim),
                                                        nn.ReLU(), 
                                                        nn.Linear(hidden_dim, data_dim)
                                                         ) for _ in range(total_timesteps)])
    def predict(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            h = self.net1(x)
            grad = self.net_t_list[i](h)
            pred += gamma * grad
        return pred
    
    def forward(self, x, time):
        # h = self.net1(torch.concat([z, x], dim=-1))
        h = self.net1(x)
        return self.net_t_list[time](h)

class ToyBoostingSep(nn.Module):          # 
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(ToyBoostingSep, self).__init__()

        self.total_timesteps = total_timesteps
        self.net1 = nn.Sequential(nn.Linear(data_dim, hidden_dim), 
                                    nn.ReLU(),
                                    # nn.Linear(hidden_dim, hidden_dim), 
                                    # nn.ReLU(),
                                    # nn.Linear(hidden_dim, hidden_dim),
                                    # nn.ReLU()
                                    )
        
        self.net_t_list = nn.ModuleList([nn.Sequential(nn.Linear(data_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim),
                                                        nn.ReLU(), 
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, data_dim)
                                                         ) for _ in range(total_timesteps)])
    def predict(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            # h = self.net1(x)
            grad = self.net_t_list[i](pred)
            pred += gamma * grad
        return pred
    
    def predict_flow(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            # h = self.net1(x)
            grad = self.net_t_list[i](pred)
            pred += gamma * grad
        return pred
    
    def forward(self, x, time):
        # h = self.net1(torch.concat([z, x], dim=-1))
        # h = self.net1(x)
        return self.net_t_list[time](x)

class ToyFlowMatching(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ToyFlowMatching, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ToyBoostingOne(nn.Module):          # 
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(ToyBoostingOne, self).__init__()

        self.total_timesteps = total_timesteps
        self.net = nn.Sequential(nn.Linear(data_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim),
                                                        nn.ReLU(), 
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, data_dim)
                                                    )
        
       
    def predict(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            grad = self.net(pred)
            pred += gamma * grad
        return pred
    
    def predict_flow(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            grad = self.net(pred)
            pred += gamma * grad
        return pred
    
    def forward(self, x):

        return self.net(x)

class ToyRegressionNet(nn.Module):   # best result - Toy-diffusion-swissroll
    """
        For predicting score of distribution (with low level sigma)
    """
    def __init__(self, data_dim=2, hidden_dim=128, num_hidden=4, device='cuda'):
        super(ToyRegressionNet, self).__init__()
        self.data_dim = data_dim
        

        # Make the network
        self.net1 = nn.Linear(data_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)


    def forward(self, x):
        """
            Forward pass of the network
        """
        x_ = self.net1(x) 
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out


class BasicUNet(nn.Module):                         # 02_diffusion_models_from_scratch
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function

        return x    


"""
GAN
"""

class ToyDiscriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
       
        self.net1 = nn.Linear(data_dim, hidden_dim)
       
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)

        return self.sigmoid(out)


class ToyGenerator(nn.Module):
    def __init__(self, data_dim, hidden_dim, z_dim):
        super().__init__()
        
        self.net1 = nn.Linear(z_dim, hidden_dim)
       
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            #  nn.ELU(alpha=0.2)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, x):

        out = self.net1(x)
        out = self.net2(out)
        out = self.net3(out)

        return out

class ToyGAN(nn.Module):
    def __init__(self, data_dim, z_dim, hidden_dim, num_hidden, device='cuda'):
        super().__init__()

        self.generator = ToyGenerator(data_dim, hidden_dim, z_dim)
        self.discriminator = ToyDiscriminator(data_dim, hidden_dim)



def select_model_diffusion(model_info, data_dim, time_dim, n_timesteps, device='cuda'):
    model_info = model_info.split('_')
    model_name = model_info[0]
    num_hidden = int(model_info[1])
    hidden_dim = int(model_info[2])

    if model_name=='ToyNetwork1':
        return ToyNetwork1(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyNetwork2':
        return ToyNetwork2(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyNetwork3':
        return ToyNetwork3(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyNetwork4':
        return ToyNetwork4(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyNetwork5':
        return ToyNetwork5(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyNetwork6':
        return ToyNetwork6(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoostingCmps':
        return ToyBoostingCmps(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)

    elif model_name=='ToyDDPM':
        return ToyDDPM(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyScoreNet':
        return ToyScoreNet(data_dim=data_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, device=device)
    elif model_name=='ToyScoreNetWithTime':
        return ToyScoreNetWithTime(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoosting':
        return ToyBoosting(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoostingTimeEmb':
        return ToyBoostingTimeEmb(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoostingSepShrd':
        return ToyBoostingSepShrd(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyRegressionNet':
        return ToyRegressionNet(data_dim=data_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, device=device)
    elif model_name=='ToyBoostingSep':
        return ToyBoostingSep(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyFlowMatching':
        return ToyFlowMatching(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoostingOne':
        return ToyBoostingOne(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)


def select_model_gan(model_info, data_dim, z_dim, device='cuda'):
    model_info = model_info.split('_')
    model_name = model_info[0]
    num_hidden = int(model_info[1])
    hidden_dim = int(model_info[2])
    
    if model_name=='ToyGAN':
        return ToyGAN(data_dim, z_dim, hidden_dim, num_hidden, device=device)
    



if __name__=='__main__':

    net1 = ToyNetwork1(time_dim=8)
    net2 = ToyNetwork2(time_dim=8)
    net3 = ToyNetwork3(time_dim=8)
    net4 = ToyNetwork4(time_dim=1)
    net5 = ToyNetwork5(time_dim=1)
    net6 = ToyNetwork6(time_dim=1)

    out1 = net1(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    out2 = net2(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    out3 = net3(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    out4 = net4(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([[10.0]], dtype=torch.float))
    out5 = net5(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    out6 = net6(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))

    unet1 = BasicUNet()
    out = unet1(torch.rand([28, 28], dtype=torch.float).view(1, 1, 28, 28))
    print()