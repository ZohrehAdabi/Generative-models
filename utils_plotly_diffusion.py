

import matplotlib.pyplot as plt
# import torch
import numpy as np
from types import SimpleNamespace
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.stats as ss
import pandas as pd
import json



def read_launch_for_terminal():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_terminal.txt', 'w') as f:
        for line in params_txt:
            if '//' in line and 'python' in line:
                line = line.replace('//', '')
                line = line.strip()
                f.write(f"{line}\n\n")

def read_launch():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_plotly.json', 'w') as f:
        for line in params_txt:
            if '//' not in line:
                f.write(line)
    with open('./.vscode/launch_plotly.json', 'r') as f:
        launch = json.load(f)
    # print(lunch.keys())
    config = launch['configurations'][2:]
    config_name_dict = {}
    for i in range(len(config)):
        config_name_dict[config[i]['name'] ] = i
    # print(config_name)
    return config, config_name_dict

def configuration_from_lunch(config, config_name_dict, config_name):
    config_params = config[config_name_dict[config_name]]['args']
    params_dict = {}
    cnfg_len = len(config_params)
    for i, p in enumerate(config_params):
        if '--' in p:
            if i < cnfg_len-1:
                if '--' not in config_params[i+1]:
                    params_dict[p[2:]] = config_params[i+1]
                else:
                    params_dict[p[2:]] = True
            else: # last params
                params_dict[p[2:]] = True
    return params_dict

def get_config(method, config_name):

    config, config_name_dict = read_launch()
    params_dict = configuration_from_lunch(config, config_name_dict, config_name)
    expr_id = f"{method}_beta_{params_dict['beta_schedule']}_T_{params_dict['n_timesteps']}_{params_dict['model']}" + \
            f"_{params_dict['dataset']}_t_dim_{params_dict['time_dim']}"
    return params_dict, expr_id

def get_params(method, expr_id):

    dataset_name, config = get_dataset_name(method, expr_id)
    params = SimpleNamespace(**config)
    return params

def get_dataset_name(method, expr_id):
    with open(f'./saved_result/{method}/configs_train/{expr_id}.json', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset']
    # num_class = 1
    # if dataset_name[0].isdigit():
    #     num_class = int(dataset_name[0])
    #     if dataset_name[1].isdigit():
    #         num_class = int(dataset_name[0:2])
     
    return dataset_name, config
    
def read_hdf_train(file, key=-1):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()[2:-1]
    # for k in keys:
    #     print(k)
    if key==-1 or key >= len(keys):
        df_sample = hdf_file[keys[-1]]
        if key > 0:
            print(f"Warning: last epoch is {int(keys[-1].split('/')[2].split('_')[2])} with key= {len(keys)}")
    else:
        df_sample = hdf_file[keys[key]]
        print(f"Epoch: {int(keys[key].split('/')[2].split('_')[2])}/{int(keys[-1].split('/')[2].split('_')[2])}")
    df_data = hdf_file['/df/data']
    n_sel_time = hdf_file['/df/info']['n_sel_time'].values[0]
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
    hdf_file.close()
    return df_sample, df_data, timesteps_list, n_sel_time, n_samples

def read_hdf(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()[2:-1]
    # for k in keys:
    #     print(k)
    df_sample = hdf_file[keys[-1]]
    df_data = hdf_file['/df/data']
    n_sel_time = hdf_file['/df/info']['n_sel_time'].values[0]
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
    if 'grad' in keys[0]:
        df_grads_or_noises = hdf_file['/df/grads']
        df_grad_or_noise_info = hdf_file['/df/grad_info']
    else:
        df_grads_or_noises = hdf_file['/df/noises']
        df_grad_or_noise_info = hdf_file['/df/noise_info']

    hdf_file.close()

    return df_sample, df_data, timesteps_list, n_sel_time, n_samples, df_grads_or_noises, df_grad_or_noise_info

def read_hdf_sample_zero_train(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()[2:]
    # for k in keys:
    #     print(k)
    df_sample = hdf_file[keys[0]]
    epoch_list = [int(keys[0].split('/')[2].split('_')[3])]
    for k in keys[1:]:
        df_sample = pd.concat([df_sample, hdf_file[k]])
        epoch_list.append(int(k.split('/')[2].split('_')[3]))
    df_data = hdf_file['/df/data']
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    hdf_file.close()
    return df_sample, df_data, epoch_list, n_samples

def prepare_df_zeros_for_plotly(df_sample, df_data, epoch_list, n_samples):

    df_sample['epoch'] = np.repeat(epoch_list, n_samples)
    df_sample['type'] = [f'sample_{i}' for i in range(1, n_samples+1)] * len(epoch_list)
    if df_data.shape[0] > 2000:
        df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    n_data = df_data.shape[0]
    df_data = pd.concat([df_data] * len(epoch_list)).reset_index(drop=True)
    df_data['epoch'] = np.repeat(epoch_list, n_data)
    df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    animation_frame = 'epoch'
    return df_sample, df_data, animation_frame

def prepare_df_for_plotly(df_sample, df_data, timesteps_list, n_sel_time):

    n_smpl = df_sample.shape[0] // n_sel_time
    
    df_sample['time'] = np.repeat(timesteps_list, n_smpl)
    df_sample['type'] = [f'sample_{i}' for i in range(1, n_smpl+1)] * n_sel_time
    if df_data.shape[0] > 2000:
        df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    n_data = df_data.shape[0]
    df_data = pd.concat([df_data] * n_sel_time).reset_index(drop=True)
    df_data['time'] = np.repeat(timesteps_list, n_data)
    df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * n_sel_time

    animation_frame = 'time'
    return df_sample, df_data, animation_frame

def prepare_plotly_fig(df_sample, df_data, animation_frame='time'):

    if animation_frame=='time':
        df_sample = df_sample[::-1].reset_index(drop=True)
    fig_smpl = px.scatter(df_sample, x='x', y='y', animation_frame=animation_frame, width=600, height=600, 
                  animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]
                  )  
    fig_data = px.scatter(df_data, x='data_x', y='data_y', animation_frame=animation_frame, width=600, height=600, 
                    animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5])
    fig_smpl['data'][0]['showlegend'] = True
    fig_data['data'][0]['showlegend'] = True

    for trace in fig_smpl.data:   # first display
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')  # ''

    for fr in fig_smpl.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')

    for trace in fig_data.data:   # first display
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')  # ''

    for fr in fig_data.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')

    return fig_smpl, fig_data

def final_plot(fig_sample, fig_data):

    fig_all = go.Figure(
        data=fig_data.data + fig_sample.data,
        frames=[
                go.Frame(data=fr_data.data + fr_smpl.data, name=fr_smpl.name)
                for fr_data, fr_smpl in zip(fig_data.frames, fig_sample.frames)
        ],
        layout=fig_sample.layout
        )
    fig_all.update_layout(title=dict(text='Sampling'), width=500, height=500, margin=dict(l=20, r=20, b=5, t=30, pad=5)) 
    fig_all['layout'].updatemenus[0]['pad'] = {'r': 10, 't': 40}
    fig_all['layout'].sliders[0]['pad'] = {'r': 10, 't': 25}
    return fig_all


def read_hdf_loss(file):

    df_loss = pd.read_hdf(file)   
    df_loss_itr = pd.DataFrame(df_loss.loc[0, ['loss_itr']].values[0], columns=['loss_itr'])
    for i in range(1, df_loss.shape[0]):
        df_loss_itr = pd.concat([df_loss_itr, pd.DataFrame(df_loss.loc[i, ['loss_itr']].values[0], columns=['loss_itr'])])
    df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    return df_loss.loc[:, ['epoch', 'loss_epoch']], df_loss_itr

def create_fig_loss(df_loss, df_loss_itr):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=df_loss_itr['loss_itr'], name="Loss itr", mode='lines', showlegend=True, line=dict(width=1, color='DarkSlateGrey')))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=(np.cumsum(df_loss_itr['loss_itr']) / df_loss_itr['itr']), name="Loss mean itr", mode='lines', showlegend=True, line=dict(width=1, color='darkviolet')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], name='Loss epoch', mode='lines', showlegend=True, line=dict(width=2, color='lightsalmon')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=(np.cumsum(df_loss['loss_epoch']) / df_loss['epoch']), name='Loss mean' ,mode='lines', line=dict(width=2, color='fuchsia')))
    fig.update_layout(title=dict(text='Loss'))
    fig.update_layout(width=800, height=320, margin=dict(l=20, r=20, b=5, t=30, pad=5))
    return fig

def create_fig_grad_noise_info(df_grad_or_noise_info, quiver_name):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    # df_grad_or_noise_info = df_grad_or_noise_info[::-1]
    fig.add_trace(go.Scatter(x=df_grad_or_noise_info['time'], y=df_grad_or_noise_info['norm'], name="Norm mean", mode='markers+lines', showlegend=True, line=dict(width=1, color='limegreen')))
    if quiver_name=='grad':
        title = 'Grad norm'
    else:
        title = 'Noise norm'
    rangex = [df_grad_or_noise_info.loc[0, 'time']+3,df_grad_or_noise_info.loc[len(df_grad_or_noise_info)-1, 'time']-2]
    fig.update_layout(title=dict(text=title))
    fig.update_layout(width=800, height=320, xaxis_range=rangex, margin=dict(l=20, r=20, b=5, t=30, pad=5))
    return fig


def prepare_plotly_fig_quiver(df_sample, df_data, n_smpl, n_sel_time, quiver_name='next', animation_frame='time'):

    if animation_frame=='time':
        df_sample = df_sample[::-1].reset_index(drop=True)

    fig_smpl = px.scatter(df_sample, x='x', y='y', animation_frame=animation_frame, width=600, height=600, 
                  animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]
                  )  
    fig_data = px.scatter(df_data, x='data_x', y='data_y', animation_frame=animation_frame, width=600, height=600, 
                    animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5])
    fig_smpl['data'][0]['showlegend'] = True
    fig_data['data'][0]['showlegend'] = True

    for trace in fig_smpl.data:   # first display
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')  # ''

    for fr in fig_smpl.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')

    for trace in fig_data.data:   # first display
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')  # ''

    for fr in fig_data.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')

    frames_quiver = []
    
    color = 'DarkSlateGrey' #'mediumvioletred' # 'deeppink'
    
    for i in range(n_sel_time):
    
        xy = df_sample.iloc[i*n_smpl:(i+1)*n_smpl, [0, 1]].reset_index(drop=True).loc[:100]
        x, y = xy['x'],  xy['y']
        uv = df_sample.iloc[(i+1)*n_smpl:(i+2)*n_smpl, [0, 1]].reset_index(drop=True).loc[:100]
        u, v = uv['x'],  uv['y']
        # print(df_sample.loc[(i)*n_smpl, 'time'], df_sample.loc[(i+1)*n_smpl, 'time'])
        if i==0:
            fig_quiver = ff.create_quiver(x, y, u-x, v-y, scale=1, name=quiver_name, arrow_scale=.5,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
            frames_quiver.append(go.Frame(data=[fig_quiver.data[0]]))
            continue

        elif i==(n_sel_time-1):
            fig_quivers = ff.create_quiver(x, x, x*0, x*0, scale=1, name=quiver_name, arrow_scale=.5,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
        else:
            fig_quivers = ff.create_quiver(x, y, u-x, v-y, scale=1, name=quiver_name, arrow_scale=.5,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
            
        frames_quiver.append(go.Frame(data=[fig_quivers.data[0]]))


    return fig_smpl, fig_data, fig_quiver, frames_quiver

def final_plot_quiver(fig_sample, fig_data, fig_quiver, frames_quiver):

    fig_all = go.Figure(
        data=fig_data.data + fig_sample.data + fig_quiver.data,
        frames=[
                go.Frame(data=fr_data.data + fr_smpl.data + fr_quiver.data, name=fr_smpl.name)
                for fr_data, fr_smpl, fr_quiver in zip(fig_data.frames, fig_sample.frames, frames_quiver)
        ],
        layout=fig_sample.layout
        )
    fig_all.update_layout(title=dict(text='Sampling'), width=500, height=500, margin=dict(l=20, r=20, b=5, t=30, pad=5)) 
    fig_all['layout'].updatemenus[0]['pad'] = {'r': 10, 't': 40}
    fig_all['layout'].sliders[0]['pad'] = {'r': 10, 't': 25}
    return fig_all

def prepare_plotly_fig_quiver_grad_or_noise(df_sample, df_data, df_grads_or_noises, 
                                            n_smpl, n_sel_time, quiver_name='grad', animation_frame='time'):

    if animation_frame=='time':
        df_sample = df_sample[::-1].reset_index(drop=True)

    fig_smpl = px.scatter(df_sample, x='x', y='y', animation_frame=animation_frame, width=600, height=600, 
                  animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]
                  )  
    fig_data = px.scatter(df_data, x='data_x', y='data_y', animation_frame=animation_frame, width=600, height=600, 
                    animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5])
    fig_smpl['data'][0]['showlegend'] = True
    fig_data['data'][0]['showlegend'] = True

    for trace in fig_smpl.data:   # first display
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')  # ''

    for fr in fig_smpl.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')

    for trace in fig_data.data:   # first display
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')  # ''

    for fr in fig_data.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')

    frames_quiver = []
    df_grads_or_noises = df_grads_or_noises[::-1].reset_index(drop=True)
    color = 'DarkSlateGrey' #'mediumvioletred' # 'deeppink'
    for i in range(n_sel_time):
    
        xy = df_sample.iloc[i*n_smpl:(i+1)*n_smpl, [0, 1]].reset_index(drop=True).loc[:100]
        x, y = xy['x'],  xy['y']
        uv = df_grads_or_noises.iloc[i*n_smpl:(i+1)*n_smpl, [0, 1]].reset_index(drop=True).loc[:100]
        u, v = uv['x'],  uv['y']
        # print(df_sample.loc[(i)*n_smpl, 'time'], df_sample.loc[(i+1)*n_smpl, 'time'])
        if i==0:
            fig_quiver = ff.create_quiver(x, y, u, v, scale=0.15, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
            frames_quiver.append(go.Frame(data=[fig_quiver.data[0]]))
            continue

        elif i==(n_sel_time-1):
            fig_quivers = ff.create_quiver(u, v, u-u, v-v, scale=0.15, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
        else:
            fig_quivers = ff.create_quiver(x, y, u, v, scale=0.15, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=2, color=color))
            
        frames_quiver.append(go.Frame(data=[fig_quivers.data[0]]))


    return fig_smpl, fig_data, fig_quiver, frames_quiver



def read_hdf_regression(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()[3:-1]
    # for k in keys:
    #     print(k)
    df_sample = hdf_file[keys[-1]]
    df_data = hdf_file['/df/data']
   
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    error = hdf_file['/df/error']['error'].values[0]

    hdf_file.close()

    return df_sample, df_data, n_samples, error

def read_hdf_prediction_regression_train(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()
    # for k in keys:
    #     print(k)
    df_prediction = pd.DataFrame()
    epoch_list = []
    error_list = []
    for k in keys:
        if 'prediction' in k:
            df_prediction = pd.concat([df_prediction if not df_prediction.empty else None, hdf_file[k]])
            epoch_list.append(int(k.split('/')[2].split('_')[2]))
        elif 'error' in k:
            error_list.append(hdf_file[k]['error'].values[0])

    df_data = hdf_file['/df/data']
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    hdf_file.close()
    return df_prediction, df_data, epoch_list, n_samples, error_list

def read_hdf_loss_regression(file):

    df_loss = pd.read_hdf(file)   
    df_loss_itr = pd.DataFrame(df_loss.loc[0, ['loss_itr']].values[0], columns=['loss_itr'])
    for i in range(1, df_loss.shape[0]):
        df_loss_itr = pd.concat([df_loss_itr, pd.DataFrame(df_loss.loc[i, ['loss_itr']].values[0], columns=['loss_itr'])])
    df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    return df_loss.loc[:, ['epoch', 'loss_epoch']], df_loss_itr

def create_fig_loss_regression(df_loss, df_loss_itr):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=df_loss_itr['loss_itr'], name="Loss itr", mode='lines', showlegend=True, line=dict(width=1, color='#AB63FA')))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=(np.cumsum(df_loss_itr['loss_itr']) / df_loss_itr['itr']), name="Loss mean itr", mode='lines', showlegend=True, line=dict(width=1, color='darkviolet')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], name='Loss epoch', mode='lines', showlegend=True, line=dict(width=2, color='lightsalmon')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=(np.cumsum(df_loss['loss_epoch']) / df_loss['epoch']), name='Loss mean' ,mode='lines', line=dict(width=2, color='fuchsia')))
    fig.update_layout(title=dict(text='Loss'))
    fig.update_layout(width=800, height=320, margin=dict(l=20, r=20, b=5, t=30, pad=5))
    return fig


def create_fig_prediction_regression(df_prediction, df_data, error):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_data['data_x'], y=df_data['data_y'], name="data", mode='markers', showlegend=True, line=dict(width=1, color='#19D3F3')))
    fig.add_trace(go.Scatter(x=df_prediction['x'], y=df_prediction['y'], name="pred", mode='markers', showlegend=True, line=dict(width=1, color='AB63FA')))
    

    fig.update_layout(title=dict(text=f'MSE: {error}'), xaxis_range=[-2.5, 2.5], yaxis_range=[-2.5, 2.5])
    fig.update_layout(width=500, height=500, margin=dict(l=20, r=20, b=20, t=30, pad=5))
    return fig

def plot_animation(method, expr_id, training=True, epoch_key=-1, all_zeros=False, quiver_name=None, test_name=None):

    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_sample_per_epoch.h5' if training else f'{path}/{expr_id}_df_sample.h5'  
    
    if test_name is not None:
       file = f'{path}/{expr_id}_df_sample_{test_name}.h5' 
    # if quiver_name is None:
    #     quiver_name = 'next'

    if training and all_zeros:
        file = f'{path}/{expr_id}_df_sample_zero_per_epoch.h5'
        df_sample, df_data, epoch_list, n_samples = read_hdf_sample_zero_train(file)
    elif training:
        df_sample, df_data, timesteps_list, n_sel_time, n_samples = read_hdf_train(file, key=epoch_key)
    else:
        df_sample, df_data, timesteps_list, n_sel_time, n_samples, df_grads_or_noises, df_grad_or_noise_info = read_hdf(file)
       

    if training and all_zeros:
        quiver_name = None
        df_sample, df_data, animation_frame = prepare_df_zeros_for_plotly(df_sample, df_data, epoch_list, n_samples)
    else:
        df_sample, df_data, animation_frame = prepare_df_for_plotly(df_sample, df_data, timesteps_list, n_sel_time)

    # fig_sample, fig_data = prepare_plotly_fig(df_sample, df_data, animation_frame)
    # fig_all = final_plot(fig_sample, fig_data)
    if quiver_name is None:
        fig_sample, fig_data = prepare_plotly_fig(df_sample, df_data, animation_frame)
        fig_all = final_plot(fig_sample, fig_data)
    elif quiver_name=='next':
        fig_sample, fig_data, fig_quiver, frames_quiver = prepare_plotly_fig_quiver(df_sample, df_data, n_samples, n_sel_time, 
                                                                                    quiver_name, animation_frame)
        fig_all = final_plot_quiver(fig_sample, fig_data, fig_quiver, frames_quiver)

    else: 
        fig_sample, fig_data, fig_quiver, frames_quiver = prepare_plotly_fig_quiver_grad_or_noise(df_sample, df_data, df_grads_or_noises,
                                                                            n_samples, n_sel_time, quiver_name, animation_frame)
    
        fig_all = final_plot_quiver(fig_sample, fig_data, fig_quiver, frames_quiver)

    fig_loss = None
    if training:
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_itr = read_hdf_loss(file)
        fig_loss = create_fig_loss(df_loss, df_loss_itr)
        return fig_all, fig_loss
    else:
        fig_grad_or_noise = create_fig_grad_noise_info(df_grad_or_noise_info, quiver_name)
        return fig_all, fig_grad_or_noise
    

def plot_animation_regression(method, expr_id, training=True, epoch_key=-1, all_zeros=False, quiver_name=None, test_name=None):

    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_prediction_per_epoch.h5' if training else f'{path}/{expr_id}_df_prediction.h5'  
    
    if test_name is not None:
       file = f'{path}/{expr_id}_df_prediction_{test_name}.h5' 
    # if quiver_name is None:
    #     quiver_name = 'next'

    if training and all_zeros:
        file = f'{path}/{expr_id}_df_prediction_zero_per_epoch.h5'
        df_prediction, df_data, epoch_list, n_samples, error_list = read_hdf_prediction_regression_train(file)
    # elif training:
    #     df_sample, df_data, timesteps_list, n_sel_time, n_samples = read_hdf_train(file, key=epoch_key)
    else:
        df_prediction, df_data, n_samples, error = read_hdf_regression(file)
       

    if training and all_zeros:
        quiver_name = None
        df_prediction, df_data, animation_frame = prepare_df_zeros_for_plotly(df_prediction, df_data, epoch_list, n_samples)
    else:
        fig_prediction = create_fig_prediction_regression(df_prediction, df_data, error)
        # df_prediction, df_data, animation_frame = prepare_df_for_plotly(df_prediction, df_data, timesteps_list, n_sel_time)

    # fig_sample, fig_data = prepare_plotly_fig(df_sample, df_data, animation_frame)
    # fig_all = final_plot(fig_sample, fig_data)
    if training and all_zeros:
        fig_sample, fig_data = prepare_plotly_fig(df_prediction, df_data, animation_frame)
        fig_all = final_plot(fig_sample, fig_data)


    fig_loss = None
    if training and all_zeros:
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_itr = read_hdf_loss_regression(file)
        fig_loss = create_fig_loss_regression(df_loss, df_loss_itr)
        return fig_all, fig_loss
    else:
        return fig_prediction, fig_loss

 


if __name__=='__main__':
    # file = r'saved_result\\2spirals\\saved_hdfs_training\\DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64_df_sample_per_epoch.h5'
    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    expr_id = 'DDPM_beta_linear_T_40_ToyDDPM_4_64_swissroll_t_dim_1'
    expr_id = 'FlowMatching_T_40_ToyFlowMatching_4_64_swissroll_t_dim_1_gamma_0.025'
    expr_id = 'Regression_ToyRegressionNet_4_64_swissroll'
    expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_500_gamma_0.025'
    expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_100_gamma_0.025'
    method = expr_id.split('_')[0]
    fig_sample, fig_loss = plot_animation(method, expr_id, training=True, all_zeros=True)
    # fig_sample, fig_loss = plot_animation_regression(method, expr_id, training=False, all_zeros=True)