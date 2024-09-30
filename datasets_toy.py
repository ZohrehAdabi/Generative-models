
import sklearn.datasets
import os
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


# All toy data lies in range [-2, 2] x [-2, 2]
DATA_NAME = ['25gaussians', '8gaussians', 'swissroll', '2spirals', '2circles', '2sines', 'checkerboard', '2moons']
#  

class Toy_Dataset(Dataset):
    def __init__(self, data_name, data_size=1000):

        self.data, _ = load_toy_data(data_name, data_size)
        # self.discrete = discrete

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # if self.discrete:
        #     raise NotImplementedError
        return self.data[idx]


def load_toy_data(dataset_name, data_size=1000):
    data_n = dataset_name.split('_')
    data_name = data_n[0]

    regr = True if data_n[-1]=='regr' else False
    labels = []
    assert data_name in DATA_NAME, "Not a proper data name"

    if regr:
        data_file = f'./toy_dataset/{data_name}_{data_size}.pkl'
        if os.path.exists(data_file):
            f = open(data_file, 'rb')
            dataset = pickle.load(f)
            f.close()        
            return dataset, labels
    
    if data_name == '25gaussians': #std = 0.05
        centers = [-1, -.5, 0, .5, 1] 
        dataset = []
        k = 0
        for x in centers:
            for y in centers:
                for i in range(data_size // 25):
                    point = np.random.randn(2) * 0.025
                    point += [x, y]
                    dataset.append(point)
                    labels.append(k)
                k += 1
        dataset = np.array(dataset, dtype='float32') * 2 #2.828
        labels = np.array(labels)

    elif data_name == '8gaussians':
        centers = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))])
        dataset = []
        
        for i in range(data_size):
            point = np.random.randn(2) * 0.025
            center_idx = np.random.choice(centers.shape[0])
            point += centers[center_idx]
            dataset.append(point)
            labels.append(center_idx)
        dataset = np.array(dataset, dtype='float32') * 2 #2.828
        labels = np.array(labels)

    elif data_name == 'swissroll':
        dataset = sklearn.datasets.make_swiss_roll(n_samples=data_size, noise=0.1)[0]
        dataset = dataset.astype('float32')[:, [0, 2]] / 6.242
        labels = np.ones(dataset.shape[0])

    elif data_name == "2spirals":
        n = np.sqrt(np.random.rand(data_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(data_size // 2, 1) * 0.01
        d1y = np.sin(n) * n + np.random.rand(data_size // 2, 1) * 0.01
        label1 = np.zeros(d1x.shape[0])
        label2 = np.ones(d1x.shape[0]) 
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
        x += np.random.randn(*x.shape) * 0.01
        dataset = x.astype('float32') / 4.828
        labels = np.vstack([label1, label2])

    elif data_name == "2circles":
        data, labels = sklearn.datasets.make_circles(n_samples=data_size, factor=.5, noise=0.0125)
        labels = (labels-1) * -1
        dataset = data.astype("float32") * 1.828

    elif data_name =="2sines":
        x = (np.random.rand(data_size) - 0.5) * (4.25) #* np.pi
        u = (2*np.random.binomial(1, 0.5, data_size) - 1) 
        y = u * np.sin(1.5*x) * 1.5 + np.random.randn(*x.shape) * 0.015
        sin1 = np.stack([x[u==-1], y[u==-1]], 1)
        sin2 = np.stack([x[u==1], y[u==1]], 1)
        label1 = np.zeros(sin1.shape[0])
        label2 = np.ones(sin2.shape[0]) 
        dataset =  np.vstack((sin1, sin2)).astype("float32") 
        labels = np.concatenate([label1, label2], axis=0)

    elif data_name =="checkerboard":
        centers = np.array([(-1.5, 1.5), (0.5, 1.5), (-0.5, 0.5), (1.5, 0.5), 
                    (-1.5, -0.5), (0.5, -0.5), (-0.5, -1.5), (1.5, -1.5)])
        dataset = []
        for i in range(data_size):
            point = np.random.uniform(-.5, .5, size=2)
            center_idx = np.random.choice(centers.shape[0])
            point += centers[center_idx]
            dataset.append(point)
            labels.append(center_idx)
        dataset = np.array(dataset, dtype='float32') #* 2
        labels = np.array(labels)

    elif data_name == "2moons":
        x = np.linspace(0, np.pi, data_size // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1)
        u += 0.01 * np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1)
        v += 0.01 * np.random.normal(size=v.shape)
        label1 = np.zeros(x.shape[0])
        label2 = np.ones(x.shape[0]) 
        dataset = np.concatenate([u, v], axis=0).astype("float32")  #* 2
        labels = np.concatenate([label1, label2], axis=0)


    if regr:
        data_file = f'./toy_datasets/{data_name}_{data_size}.pkl'
        p = Path(f'./toy_datasets')
        p.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(data_file):
            f = open (data_file, 'wb')
            pickle.dump(dataset, f)
            f.close()

    return dataset, labels


# -------- Utils for Toy plotting -------- #
def plot_toy(data, path):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    plt.clf()
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(data[:, 0], data[:, 1], c='cornflowerblue', marker='.', s=5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    plt.savefig(path)


def select_dataset(dataset_name, batch_size, total_size, normalize=False, device='cuda'):

    data, label = load_toy_data(dataset_name, data_size=total_size)
    # dataset = Toy_Dataset(dataset_name, data_size=total_size)
    scaler = None
    if normalize:
        # dataset = (data - data.mean()) / data.std()
        # scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        dataset = scaler.transform(data)
    else:
        dataset = data
    dataset = torch.from_numpy(dataset).to(torch.float).to(device)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, data, scaler

def get_noise_dataset(total_size, batch_size, device='cuda'):

    niose_file = f'./toy_datasets/niose_{total_size}.pkl'
    p = Path(f'./toy_datasets')
    p.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(niose_file):
        f = open (niose_file, 'wb')
        noise_dataset = np.random.randn(total_size, 2)
        pickle.dump(noise_dataset, f)
        f.close()
    else:
        f = open(niose_file, 'rb')
        noise_dataset = pickle.load(f)
        f.close()  
    noise_dataset = torch.from_numpy(noise_dataset).to(torch.float).to(device)    
    dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


if __name__ == '__main__':
    # plot_toy(next(load_noise(2048)), 'noise.png')
    # for data_name in DATA_NAME:
    #     dataset = Toy_Dataset(data_name, data_size=2048)
    #     dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    #     plot_toy(next(iter(dataloader)), f"{data_name}.png")
    #     print()
    # dataset = Toy_Dataset('25gaussians', data_size=4*24)
    # dataloader = DataLoader(dataset, batch_size=24, shuffle=True)
    # for x in dataloader:
    #     print(x)
    # dataset = Toy_Dataset('2moons', data_size=2048) 
    select_dataset('2moons', 10, 100)
    print()

