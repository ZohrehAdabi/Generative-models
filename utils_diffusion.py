from pathlib import Path
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


def plot_samples(samples, data, path):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    if data.shape[0] > 2000:
        data = data[np.linspace(0, data.shape[0]-1, 2000, dtype=int), :]
    plt.clf()
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
        Xvis = intermediate_smpl[i].cpu()
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker=".", color='m', animated=True)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        return fig,

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=True)
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
                    'beta_schedule': params.beta_schedule, 
                    'n_timesteps': params.n_timesteps, 
                    'model': params.model,
                    'dataset': params.dataset,
                    'time_dim': params.time_dim,
                    'data_dim': params.data_dim,
                    'batch_size': params.batch_size,
                    'total_size': params.total_size,
                    'normalize': params.normalize,
                    'n_epoch': params.n_epoch,
                    'lr': params.lr,
                    'seed': params.seed
                    }
    
    if params.method == 'BoostingOne':
        config_dict['pred_goal'] =  params.pred_goal
        config_dict['gamma'] =  params.gamma
    if params.method == 'Boosting':
        config_dict['learner_inp'] =  params.learner_inp
        config_dict['innr_epoch'] =  params.innr_epoch
        config_dict['grad_type'] =  params.grad_type
        config_dict['gamma'] =  params.gamma

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

def create_hdf_file_training(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs_training' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_loss = str(path_save) + f'/{expr_id}_df_loss_per_epoch.h5'
    file_sample = str(path_save) + f'/{expr_id}_df_sample_per_epoch.h5'
    file_sample_zero_all = str(path_save) + f'/{expr_id}_df_sample_zero_per_epoch.h5'
    df_loss_per_epoch = pd.DataFrame(columns=['epoch', 'itr', 't', 'loss_itr', 'loss_epoch'])
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 1000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 1000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])

    sampling_timesteps = select_steps(n_timestep_smpl, n_sel_time)
    
    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time], 'n_samples': [n_samples]}), format='t')
        hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')

    with pd.HDFStore(file_sample_zero_all, 'w') as hdf_store_samples_zero:
        hdf_store_samples_zero.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples_zero.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples], 'n_sel_time': [n_sel_time]}), format='t')

    return file_loss, file_sample,file_sample_zero_all, df_loss_per_epoch

def create_hdf_file(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_sample = str(path_save) +  (f'/{expr_id}_df_sample.h5' if test_name is None else f'/{expr_id}_df_sample_{test_name}.h5')
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])
    # if dfs.shape[0] > 2000:
    #     dfs = dfs.loc[np.linspace(0, dfs.shape[0]-1, 2000, dtype=int), :]
    
    sampling_timesteps = select_steps(n_timestep_smpl+1, n_sel_time)

    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time], 'n_samples': [n_samples]}), format='t')
        hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')


    return file_sample

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


def create_save_dir_regression_training(save_dir, expr_id):

    p = Path(save_dir)
    
    path_save = p / 'saved_models'
    path_save.mkdir(parents=True, exist_ok=True)
    path_save = p / 'plot_predictions_training' / expr_id
    path_save.mkdir(parents=True, exist_ok=True)

def create_save_dir_regression(save_dir):

    p = Path(save_dir)

    path_save = p / 'plot_predictions'
    path_save.mkdir(parents=True, exist_ok=True)

def create_hdf_file_regression_training(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs_training' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_loss = str(path_save) + f'/{expr_id}_df_loss_per_epoch.h5'
    file_prediction = str(path_save) + f'/{expr_id}_df_prediction_per_epoch.h5'
    
    df_loss_per_epoch = pd.DataFrame(columns=['epoch', 'itr', 'loss_itr', 'loss_epoch'])
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])

    
    
    with pd.HDFStore(file_prediction, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples]}), format='t')
        

    return file_loss, file_prediction, df_loss_per_epoch

def create_hdf_file_regression(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_prediction = str(path_save) +  (f'/{expr_id}_df_prediction.h5' if test_name is None else f'/{expr_id}_df_prediction_{test_name}.h5')
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])
    # if dfs.shape[0] > 2000:
    #     dfs = dfs.loc[np.linspace(0, dfs.shape[0]-1, 2000, dtype=int), :]
    
    
    with pd.HDFStore(file_prediction, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time], 'n_samples': [n_samples]}), format='t')
        


    return file_prediction


def extract_at_time_t(var, t):
    
    selected = torch.gather(var, 0, t)
    reshape = [t.shape[0], -1] 
    return selected.reshape(*reshape)#.to(x.device)

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