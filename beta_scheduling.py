import torch
import numpy as np 


def cosine_beta_schedule(n_timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps)
    # x = torch.arange(0, n_timesteps, dtype=torch.float32)
    schedule = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = schedule / schedule[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # betas = 1 - alphas_cumprod / torch.concatenate([alphas_cumprod[0:1], alphas_cumprod[0:-1]])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(beta_start=0.0001, beta_end=0.02, n_timesteps=100):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, n_timesteps)

def quadratic_beta_schedule(n_timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start**0.5, beta_end**0.5, n_timesteps) ** 2

def sigmoid_beta_schedule(n_timesteps, beta_start=0.0001, beta_end=0.02):
    beta_start = beta_start
    beta_end = beta_end
    betas = torch.linspace(-6, 6, n_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start




def select_beta_schedule(s, beta_start=0.0001, beta_end=0.02, n_timesteps=100):
    N = n_timesteps
    if s=='constant':
        return torch.ones(N) * 0.004
    elif s=='linear':
        return linear_beta_schedule(beta_start, beta_end, N)
    elif s=='quadratic':
        return quadratic_beta_schedule(beta_start, beta_end, N)
    elif s=='sigmoid':
        return sigmoid_beta_schedule(beta_start, beta_end, N)
    elif s=='cosine':
        return cosine_beta_schedule(N)
    
if __name__=='__main__':
    import plotly.graph_objects as go
    
    fig = go.Figure()
    schedule = ['constant', 'linear', 'quadratic', 'sigmoid', 'cosine']

    N= 1000
    for s in schedule:
        beta = select_beta_schedule(s).numpy()
        x = np.arange(beta.shape[0])
        fig.add_trace(go.Scatter(x=x, y=beta, mode='lines', name=schedule[s-1]))
    fig.update_layout(title=r'$\beta_t \; schedulings$')
    fig.show()