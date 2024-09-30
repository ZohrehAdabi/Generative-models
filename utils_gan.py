from pathlib import Path
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


def save_plot_fig_stats(stats, labels, colors, title, file_name, xmax=None):

    def plot_fig_loss(stats_hist, label, color):
        plt.plot(stats_hist, label=label, color=color) # marker='o', linestyle='dashed',linewidth=2, markersize=12)
    plt.figure()
    for stats, l, c in zip(stats, labels, colors):
        plot_fig_loss(stats, l, c)
    if xmax is not None: plt.hlines(y=.5, xmin=0, xmax=xmax, colors='gray', linestyles='--', lw=2, label=r'$D^*$')
    plt.legend()
    plt.title(title)
    plt.savefig(file_name)
    plt.close()

def plot_samples(samples, data, path):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    if data.shape[0] > 2000:
        data = data[np.linspace(0, data.shape[0]-1, 2000, dtype=int), :]
    plt.figure(dpi=150)
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(data[:, 0], data[:, 1], label='data', c='C9', marker='.', s=20)
    plt.scatter(samples[:, 0], samples[:, 1], label='sample', c='C4', marker='.', s=20)
     
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    plt.legend()
    plt.savefig(path)
    plt.close()

def save_animation(intermediate_smpl, path):
    def draw_frame(i):
        plt.clf()
        Xvis = intermediate_smpl[i]
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker=".", color='m', animated=True)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        return fig,

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=500)
    anim.save(path, writer=writer)
    # anim.save(path, fps=10)

def save_animation_quiver(intermediate_smpl, intermediate_grad_g_z, dataset, grid_x, grid_y, epoch, path):

    def draw_frame(i):
        plt.clf()
        plt.scatter(dataset[:, 0], dataset[:, 1], label='data', c='C9', marker='.', s=20, animated=True)
        Xvis = intermediate_smpl[i]
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1],  label='sample', marker=".", color='C4', animated=True)
        # plt.xlim([-2.5, 2.5])
        # plt.ylim([-2.5, 2.5])
        grad_g_z = intermediate_grad_g_z[i]
        u, v = grad_g_z[:, 0], grad_g_z[:, 1]
        magnitude = np.hypot(u, v)
        # plt.scatter(self.grid_x, self.grid_y, color='k', s=1)

        
        plt.quiver(grid_x, grid_y, u, v, magnitude, scale=None, cmap='plasma', pivot='tail', angles='xy', units='width', animated=True) 
        plt.title(f'epoch: {epoch+1:<6}, t: {i} | '+r'$\frac{\partial{D(G(z))}}{\partial{G(z)}}$')
        # plt.close()
        return fig,

    fig = plt.figure(dpi=150)
    # grad_g_z = intermediate_grad_g_z[0]
    # u, v = grad_g_z[:, 0], grad_g_z[:, 1]
    # magnitude = np.hypot(u, v)
    # plt.quiver(grid_x, grid_y, u, v, magnitude, scale=0.025, cmap='plasma', pivot='tail', angles='xy', units='width', animated=True) 
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=True)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save(path, writer=writer)
    anim.save(path, fps=10)

def evaluate_scores():

    # inception_score
    # frechet_inception_distance
    # mode_score
    # rkl_divergence
    # missing modes
    
    pass

def save_config_json(save_dir, params, expr_id):

    save_config = f'{save_dir}/configs_train/'
    p = Path(save_config)
    p.mkdir(parents=True, exist_ok=True)
    save_config = str(p / f'{expr_id}.json')
    config_dict = {'save_dir': save_dir,
                   # 'method': params.method,
                    'model': params.model,
                    'dataset': params.dataset,
                    'z_dim': params.z_dim,
                    'data_dim': params.data_dim,
                    'loss_dsc': params.loss_dsc,
                    'loss_gen': params.loss_gen,
                    'batch_size': params.batch_size,
                    'total_size': params.total_size,
                    'normalize': params.normalize,
                    'n_epoch': params.n_epoch,
                    'lr_dsc': params.lr_dsc,
                    'lr_gen': params.lr_gen,
                    'seed': params.seed
                    }
    if params.method == 'GAN-wo-G':
        config_dict['lr_fk_dt'] =  params.lr_fk_dt
        
    with open(save_config, 'w') as outfile:
        json.dump(config_dict, outfile, indent=4)


def create_save_dir_training(save_dir, expr_id):

    p = Path(save_dir)
    
    path_save = p / 'saved_models'
    path_save.mkdir(parents=True, exist_ok=True)
    path_save = p / 'plot_samples_training' / expr_id
    path_save.mkdir(parents=True, exist_ok=True)

def create_save_dir(save_dir):

    p = Path(save_dir)

    path_save = p / 'plot_samples'
    path_save.mkdir(parents=True, exist_ok=True)

def create_hdf_file_training(save_dir, expr_id, dataset, n_samples, grid_size, n_sel_time):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs_training' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_loss = str(path_save) + f'/{expr_id}_df_loss_per_epoch.h5'
    file_sample = str(path_save) + f'/{expr_id}_df_sample_per_epoch.h5'
    
    df_loss_score_per_epoch = pd.DataFrame(columns=['epoch', 'itr', 'loss_dsc_itr', 'loss_gen_itr', 'loss_fk_dt_itr',
                                                     'loss_dsc_epoch', 'loss_gen_epoch', 'loss_fk_dt_epoch',
                                                    'real_score_epoch', 'fake_score_epoch', 'fake_data_score_epoch'])
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 1000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 1000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])

    
    
    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples], 'grid_size':[grid_size], 'n_sel_time':[n_sel_time]}), format='t')
        

    return file_loss, file_sample, df_loss_score_per_epoch

def create_hdf_file(save_dir, expr_id, dataset, n_samples, grid_size, n_sel_time, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_sample = str(path_save) +  (f'/{expr_id}_df_sample.h5' if test_name is None else f'/{expr_id}_df_sample_{test_name}.h5')
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])

    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples], 'grid_size':[grid_size], 'n_sel_time':[n_sel_time]}), format='t')
        
    return file_sample

def get_fake_sample_grid(grid_size=10):
    
    xs = torch.linspace(-2.5, 2.5, steps=grid_size)
    ys = torch.linspace(-2.5, 2.5, steps=grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    fake_images = torch.tensor([])
    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        f_img = torch.hstack([x.view(-1, 1), y.view(-1, 1)])
        fake_images = torch.cat([fake_images, f_img])
    return fake_images, grid_x, grid_y

def select_steps(n_timestep_smpl, n_sel_time):
    
    steps = [0]
    num_tim_stp = n_timestep_smpl #intermediate_smpl.shape[0]
    prcntg_reducing = num_tim_stp // n_sel_time
    start = 1
    [steps.append(t) for t in np.linspace(start, num_tim_stp-2, n_sel_time-2, dtype=int)]
    steps.append(num_tim_stp-1)
    

    return steps

def select_samples_for_plot(intermediate_smpl, n_timestep_smpl, n_sel_time):
    

    # if intermediate_smpl.shape[0] == n_timestep_smpl:
    #     n_timestep_smpl -= 1
    n_timestep_smpl = intermediate_smpl.shape[0]
    steps = select_steps(n_timestep_smpl, n_sel_time)
    selected = intermediate_smpl[steps, :, :].reshape(-1, intermediate_smpl.shape[-1])
    return selected


def read_launch_for_terminal():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_terminal.txt', 'w') as f:
        for line in params_txt:
            if '//' in line and 'python' in line:
                line = line.replace('//', '')
                line = line.strip()
                f.write(f"{line}\n\n")

if __name__=='__main__':

    # from beta_scheduling import select_beta_schedule
    # n_timesteps = 10
    # beta = select_beta_schedule('linear', n_timesteps=n_timesteps).to('cuda')
    # x = torch.tensor(np.random.randn(20).reshape(-1, 2)).to('cuda')
    # t = torch.randint(low=0, high=n_timesteps, size=[x.shape[0]]).to('cuda')
    # bt = extract_at_time_t(beta, t)
    print()
    read_launch_for_terminal()