

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

from utils_gan import get_fake_sample_grid

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
    


def read_hdf(file):

    hdf_file = pd.HDFStore(file)   
    keys = hdf_file.keys()[3:-1]

    df_sample = hdf_file[keys[-1]]
    df_data = hdf_file['/df/data']
   
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    df_score_info = hdf_file['/df/score_data_sample_info']

    hdf_file.close()

    return df_sample, df_data, df_score_info, n_samples

def read_hdf_train(file):

    try:
        hdf_file = pd.HDFStore(file)   
        # print(smple.keys())
        keys = hdf_file.keys()
        # for k in keys:
        #     print(k)
        df_sample = pd.DataFrame()
        df_score_info = pd.DataFrame()

        epoch_list = []
        for k in keys:
            if 'samples' in k:         
                epoch_list.append(int(k.split('/')[2].split('_')[2]))
                key_sample = k
            elif 'score_data_sample_info' in k:
                key_score = k
            elif 'grad_g_z' in k:
                key_grad = k

        df_sample = hdf_file[key_sample]   
        df_score_info = hdf_file[key_score]
        df_grad_g_z = hdf_file[key_grad]
        df_data = hdf_file['/df/data']
        n_samples = hdf_file['/df/info']['n_samples'].values[0]
        hdf_file.close()

    except:
        print(f'Error in reading hdf file {file}')
        hdf_file.close()
        
    return df_sample, df_data, df_score_info, df_grad_g_z, n_samples

def read_hdf_intermediate_train(file, epoch_key):

    try:
        hdf_file = pd.HDFStore(file)   
        # print(smple.keys())
        keys = hdf_file.keys()
        # for k in keys:
        #     print(k)
        df_intermediate_sample = pd.DataFrame()
        df_score_hist_info = pd.DataFrame()
        df_intermediate_grad_g_z = pd.DataFrame()

        epoch_list = []
        for k in keys:
            if 'intermediate_smpl' in k:         
                epoch_list.append(int(k.split('/')[2].split('_')[3]))

        if epoch_key > len(epoch_list):
            print(f"Warning: last epoch is {epoch_list[-1]} with key= {len(epoch_list)}")
        else:
            print(f"Epoch: {epoch_list[epoch_key]}/{epoch_list[-1]}")

        key_sample = f'df/intermediate_smpl_epoch_{epoch_list[epoch_key]:06}'
        key_score = f'df/score_data_sample_hist_{epoch_list[epoch_key]:06}'
        key_grad = f'df/intermediate_grad_g_z_{epoch_list[epoch_key]:06}'
        df_intermediate_sample = hdf_file[key_sample]   
        df_score_hist_info = hdf_file[key_score]
        df_intermediate_grad_g_z = hdf_file[key_grad]
        df_data = hdf_file['/df/data']
        n_samples = hdf_file['/df/info']['n_samples'].values[0]
        grid_size = hdf_file['/df/info']['grid_size'].values[0]
        n_sel_time = hdf_file['/df/info']['n_sel_time'].values[0]
        hdf_file.close()

    except:
        print(f'Error in reading hdf file {file}')
        hdf_file.close()
        
    return df_intermediate_sample, df_data, df_score_hist_info, df_intermediate_grad_g_z, n_samples, grid_size, n_sel_time

def read_hdf_all_epochs_train(file):

    try:
        hdf_file = pd.HDFStore(file)   
        # print(smple.keys())
        keys = hdf_file.keys()
        # for k in keys:
        #     print(k)
        df_sample = pd.DataFrame()
        df_score_info = pd.DataFrame()
        df_grad_g_z = pd.DataFrame()

        df_fake_data = pd.DataFrame()
        df_intermediate_smpl = pd.DataFrame()
        epoch_list = []
        for k in keys:
            if 'samples' in k:
                df_sample = pd.concat([df_sample if not df_sample.empty else None, hdf_file[k]])
                epoch_list.append(int(k.split('/')[2].split('_')[2]))
            elif 'score_data_sample_info' in k:
                df_score_info = pd.concat([df_score_info if not df_score_info.empty else None, hdf_file[k]])
            elif 'grad_g_z' in k:
                df_grad_g_z = pd.concat([df_grad_g_z if not df_grad_g_z.empty else None, hdf_file[k]])
            elif 'fake_data' in k:
                df_grad_g_z = pd.concat([df_fake_data if not df_fake_data.empty else None, hdf_file[k]])

        df_data = hdf_file['/df/data']
        n_samples = hdf_file['/df/info']['n_samples'].values[0]
        grid_size = hdf_file['/df/info']['grid_size'].values[0]
        hdf_file.close()
    except:
        print(f'Error in reading hdf file {file}')
        hdf_file.close()

    return df_sample, df_data, df_score_info.reset_index(drop=True), df_grad_g_z, epoch_list, n_samples, grid_size

def read_hdf_loss(file):

    df_loss = pd.read_hdf(file)   
    df_loss_dsc_itr = pd.DataFrame(df_loss.loc[0, ['loss_dsc_itr']].values[0], columns=['loss_dsc_itr'])
    df_loss_gen_itr = pd.DataFrame(df_loss.loc[0, ['loss_gen_itr']].values[0], columns=['loss_gen_itr'])
    
    for i in range(1, df_loss.shape[0]):
        df_loss_dsc_itr = pd.concat([df_loss_dsc_itr, pd.DataFrame(df_loss.loc[i, ['loss_dsc_itr']].values[0], columns=['loss_dsc_itr'])])
        df_loss_gen_itr = pd.concat([df_loss_gen_itr, pd.DataFrame(df_loss.loc[i, ['loss_gen_itr']].values[0], columns=['loss_gen_itr'])])
    df_loss_dsc_itr['itr'] = np.arange(df_loss_dsc_itr.shape[0])
    df_loss_gen_itr['itr'] = np.arange(df_loss_gen_itr.shape[0])

    df_score = df_loss.loc[:, ['epoch', 'real_score_epoch', 'fake_score_epoch']]
    return df_loss.loc[:, ['epoch', 'loss_dsc_epoch', 'loss_gen_epoch']], df_loss_dsc_itr, df_loss_gen_itr, df_score

def prepare_df_all_epochs_for_plotly(df_sample, df_data, df_grad_g_z, df_score_info, epoch_list, n_samples, grid_size):

    df_sample['epoch'] = np.repeat(epoch_list, n_samples)
    df_sample['type'] = [f'sample_{i}' for i in range(1, n_samples+1)] * len(epoch_list)
    if df_data.shape[0] > 2000:
        df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    n_data = df_data.shape[0]
    df_data = pd.concat([df_data] * len(epoch_list)).reset_index(drop=True)
    df_data['epoch'] = np.repeat(epoch_list, n_data)
    df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    # n_data = df_fake_sample.shape[0]
    # df_fake_sample = pd.concat([df_fake_sample] * len(epoch_list)).reset_index(drop=True)
    # df_fake_sample['epoch'] = np.repeat(epoch_list, n_data)
    # df_fake_sample['type'] = [f'grid_sample_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    df_grad_g_z['epoch'] = np.repeat(epoch_list, grid_size**2)
    df_score_info['epoch'] = epoch_list

    animation_frame = 'epoch'
    return df_sample, df_data, df_grad_g_z, df_score_info, animation_frame

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

def final_plot(fig_sample, fig_data, method):

    fig_all = go.Figure(
        data=fig_data.data + fig_sample.data,
        frames=[
                go.Frame(data=fr_data.data + fr_smpl.data, name=fr_smpl.name)
                for fr_data, fr_smpl in zip(fig_data.frames, fig_sample.frames)
        ],
        layout=fig_sample.layout
        )
    fig_all.update_layout(title=dict(text=f'{method} Sampling'), width=500, height=500, margin=dict(l=20, r=20, b=5, t=30, pad=5)) 
    fig_all['layout'].updatemenus[0]['pad'] = {'r': 10, 't': 40}
    fig_all['layout'].sliders[0]['pad'] = {'r': 10, 't': 25}
    return fig_all


def final_plot_quiver(fig_sample, fig_data, fig_quiver, frames_quiver, df_score_info, method, training=False):

    for fr in range(len(fig_sample.frames)):

        fig_sample.frames[fr]['layout'].update(title_text=f"Sample => D(x): {df_score_info.loc[fr, 'data_score_mean']:.4f} D(G(z)): {df_score_info.loc[fr, 'sample_score_mean']:.4f} ", 
                                            title_font_size=14)
    fig_all = go.Figure(
        data=fig_data.data + fig_sample.data + fig_quiver.data,
        frames=[
                go.Frame(data=fr_data.data + fr_smpl.data + fr_quiver.data, name=fr_smpl.name)
                for fr_data, fr_smpl, fr_quiver in zip(fig_data.frames, fig_sample.frames, frames_quiver)
        ],
        layout=fig_sample.layout
        )
    
    title = f'{method} Training' if training else f'{method} Sampling'
    fig_all.update_layout(title=dict(text=title), width=500, height=500, margin=dict(l=20, r=20, b=5, t=30, pad=5)) 
    
    fig_all['layout'].updatemenus[0]['pad'] = {'r': 10, 't': 40}
    fig_all['layout'].sliders[0]['pad'] = {'r': 10, 't': 25}
    return fig_all

def prepare_plotly_fig_quiver_grad(df_sample, df_data, df_fake_sample, df_grads, epoch_list, 
                                            grid_size, quiver_name='grad', animation_frame='time'):

    if animation_frame=='time':
        df_sample = df_sample[::-1]
    df_sample = df_sample.reset_index(drop=True)
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
    if animation_frame=='time':
        df_grads = df_grads[::-1]
    df_grads = df_grads.reset_index(drop=True)
    df_fake_sample = df_fake_sample.reset_index(drop=True)
    color = 'DarkSlateGrey' #'mediumvioletred' # 'deeppink'
    scale = 1
    xy = df_fake_sample
    x, y = xy['x'],  xy['y']
    for i in epoch_list:

        uv = df_grads.loc[df_grads.epoch==i]
        u, v = uv['u'],  uv['v']
        # u, v = u.reshape(-1, grid_size), v.reshape(-1, grid_size)
        scale = np.sqrt(np.linalg.norm(x) / np.linalg.norm(u)) + np.sqrt(np.linalg.norm(y) / np.linalg.norm(v))
        if i==epoch_list[0]:
            fig_quiver = ff.create_quiver(x, y, u, v, scale=scale, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=1, color=color))
            frames_quiver.append(go.Frame(data=[fig_quiver.data[0]]))
            continue

        elif i==(len(epoch_list)-1):
            fig_quivers = ff.create_quiver(x, y, u, v, scale=scale, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=1, color=color))
        else:
            fig_quivers = ff.create_quiver(x, y, u, v, scale=scale, name=quiver_name, arrow_scale=.2,
                        angle=15*np.pi/180, line=dict(width=1, color=color))
            
        frames_quiver.append(go.Frame(data=[fig_quivers.data[0]]))


    return fig_smpl, fig_data, fig_quiver, frames_quiver



def create_fig_loss(df_loss, df_loss_dsc_itr, df_loss_gen_itr, df_score):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    x = df_loss_dsc_itr['itr']
    fig.add_trace(go.Scatter(x=x, y=df_loss_dsc_itr['loss_dsc_itr'], name="Loss D", mode='lines', visible=True, showlegend=True, line=dict(width=1, color='limegreen')))
    fig.add_trace(go.Scatter(x=x, y=df_loss_gen_itr['loss_gen_itr'], name="Loss G", mode='lines', visible=True, showlegend=True, line=dict(width=1, color='crimson'))) 
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss_dsc_itr['loss_dsc_itr']) / x), name="Loss D mean", mode='lines', visible=True, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss_gen_itr['loss_gen_itr']) / x), name="Loss G mean", mode='lines', visible=True, showlegend=True, line=dict(width=2, color='deeppink')))
    x = df_loss['epoch']
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_dsc_epoch'], name='Loss D', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen'))) # lightsalmon
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_gen_epoch'], name='Loss G', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='crimson'))) # indianred
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss['loss_dsc_epoch']) / x), name='Loss D mean' , mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss['loss_gen_epoch']) / x), name='Loss G mean' ,mode='lines', visible=False, showlegend=True, line=dict(width=2, color='deeppink')))
    fig.update_layout(title=dict(text='Loss'))

    
    fig.add_trace(go.Scatter(x=x, y=df_score['real_score_epoch'], name='D(x)', mode='lines',visible=False, showlegend=True, line=dict(width=1, color='darkturquoise')))
    fig.add_trace(go.Scatter(x=x, y=df_score['fake_score_epoch'], name='D(G(z))', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='darkorange')))
    y_real_score_mean = np.cumsum(df_score['real_score_epoch'])/x
    y_fake_score_mean = np.cumsum(df_score['fake_score_epoch'])/x
   
    # plot std
    num_epoch = x.shape[0]
    def std_mean_for_score(df, y_score_mean):
        y = df
        std_mean = np.empty([num_epoch])
        for i in range(num_epoch):
            std_mean[i] = np.std(y.iloc[:i+1])
        y_upper, y_lower = y_score_mean + std_mean, y_score_mean - std_mean
        y_with_std = pd.concat([y_upper, y_lower[::-1].reset_index(drop=True)]).reset_index(drop=True)
        return y_with_std
    x = df_score['epoch']
    y_real_with_std = std_mean_for_score(df_score['real_score_epoch'], y_real_score_mean)    
    y_fake_with_std = std_mean_for_score(df_score['fake_score_epoch'], y_fake_score_mean)
    x_x_revr = pd.concat([x, x[::-1].reset_index(drop=True)]).reset_index(drop=True)
    fig.add_trace(go.Scatter(x=x_x_revr, y=y_real_with_std, fill='toself', name='D(x) std',  visible=False, showlegend=True, fillcolor='rgba(0,176,246,0.2)', line_color='rgba(255,255,255,0)'))
    fig.add_trace(go.Scatter(x=x_x_revr, y=y_fake_with_std, fill='toself', name='D(G(z)) std',  visible=False, showlegend=True, fillcolor='rgba(231,107,243,0.2)', line_color='rgba(255,255,255,0)'))

    fig.add_trace(go.Scatter(x=x, y=y_real_score_mean, name='D(x) mean', mode='lines',visible=False, showlegend=True, line=dict(width=2, color='deepskyblue'))) #dodgerblue darkturquoise deepskyblue
    fig.add_trace(go.Scatter(x=x, y=y_fake_score_mean, name='D(G(z)) mean', mode='lines', visible=False, showlegend=True, line=dict(width=2, color='orangered'))) # indianred
 
    fig.add_trace(go.Scatter(x=x, y=np.ones(num_epoch) * 0.5, name='D*', mode='lines', visible=False, showlegend=True, line=dict(width=2, color='slategray', dash='dash')))
    # fig.add_hline(y=0.5, line_width=2, line_dash="dash", name='D*', line_color="slategray", visible=False, showlegend=True)
    # hline = [{'line': {'color': 'slategray', 'dash': 'dash', 'width': 2},
    #                        'name': 'D*',
    #                        'showlegend': True,
    #                        'type': 'line',
    #                        'visible': False,
    #                        'x0': 0,
    #                        'x1': 1,
    #                        'xref': 'x domain',
    #                        'y0': 0.5,
    #                        'y1': 0.5,
    #                        'yref': 'y'}]
    updatemenus = [dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'Loss itr',
                 method = 'update',
                 args = [{'visible': [True] * 4 + [False] * 4 + [False] * 4 + [False] * 3},
                         {'title': 'Loss itr'}]),
            dict(label = 'Loss epoch',
                 method = 'update',
                 args = [{'visible': [False] * 4 + [True] * 4 + [False] * 4 + [False] * 3},
                         {'title': 'Loss epoch'}]),
            dict(label = 'Score',
                 method = 'update',
                 args = [{'visible': [False] * 8 + [True] * 2 + [False] * 4 + [True]},
                         {'title': 'Score'}]) ,
            dict(label = 'Score mean_std',
                 method = 'update',
                 args = [{'visible': [False] * 8 + [False] * 2 + [True] * 5},
                         {'title': 'Score'}]) ,
            # dict(label="None",
            #         method="relayout",
            #         args=["shapes", hline]),
                         ]) ) ]

    fig.update_layout(updatemenus=updatemenus)
    fig.update_layout(width=800, height=320, margin=dict(l=100, r=20, b=5, t=30, pad=5))
    return fig


def create_fig_sample(df_sample, df_data, df_score_info):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_data['data_x'], y=df_data['data_y'], name="data", mode='markers', showlegend=True, line=dict(width=1, color='#19D3F3')))
    fig.add_trace(go.Scatter(x=df_sample['x'], y=df_sample['y'], name="sample", mode='markers', showlegend=True, line=dict(width=1, color='#AB63FA')))
   
    fig.update_layout(title=dict(text=f"D(x): {df_score_info['data_score_mean'][0]:<4.3f} ±{df_score_info['data_score_std'][0]:<4.3f}<br>D(G(z)): {df_score_info['sample_score_mean'][0]:<4.3f} ±{df_score_info['sample_score_std'][0]:<4.3f}"), xaxis_range=[-2.5, 2.5], yaxis_range=[-2.5, 2.5])
    fig.update_layout(width=500, height=500, margin=dict(l=30, r=10, b=15, t=50, pad=1))
    return fig
   

def plot_animation(method, expr_id, training=True, epoch_key=-1, all_epochs=False, quiver_name=None, test_name=None):

    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_sample_per_epoch.h5' if training else f'{path}/{expr_id}_df_sample.h5'  
    method_type = f"{method} [{config['loss_dsc']} {config['loss_gen']}]"
    if test_name is not None:
       file = f'{path}/{expr_id}_df_sample_{test_name}.h5' 
    # if quiver_name is None:
    #     quiver_name = 'next'

    if training and all_epochs:
        df_sample, df_data,  df_score_info, df_grad_g_z, epoch_list, n_samples, grid_size = read_hdf_all_epochs_train(file)
    elif training:
        if epoch_key == -1:
            df_sample, df_data,  df_score_info, df_grad_g_z, n_samples = read_hdf_train(file) # last epoch
        else:
            df_intermediate_sample, df_data, df_score_hist_info, df_intermediate_grad_g_z,\
                  n_samples, grid_size, n_sel_time = read_hdf_intermediate_train(file, epoch_key)
    else:
        df_sample, df_data, df_score_info, n_samples = read_hdf(file)
       

    if training and all_epochs:
            fake_samples, grid_x, grid_y =get_fake_sample_grid(grid_size)
            df_fake_sample = pd.DataFrame(fake_samples, columns=['x', 'y'])
            df_sample, df_data, df_grad_g_z, df_score_info, animation_frame = \
                prepare_df_all_epochs_for_plotly(df_sample, df_data, df_grad_g_z, df_score_info, epoch_list, n_samples, grid_size)       
    elif training and epoch_key!=-1:
        fake_samples, grid_x, grid_y =get_fake_sample_grid(grid_size)
        df_fake_sample = pd.DataFrame(fake_samples, columns=['x', 'y'])
        df_sample, df_data, df_grad_g_z, df_score_info, animation_frame = \
            prepare_df_all_epochs_for_plotly(df_sample, df_data, df_grad_g_z, df_score_info, n_samples, grid_size)       

    else:
        fig_all = create_fig_sample(df_sample, df_data, df_score_info)
        

    if training and all_epochs:

        if quiver_name is None:
            fig_sample, fig_data = prepare_plotly_fig(df_sample, df_data, animation_frame)
            fig_all = final_plot(fig_sample, fig_data, method=method_type)
        else:
            fig_sample, fig_data, fig_quiver, frames_quiver = prepare_plotly_fig_quiver_grad(df_sample, df_data, df_fake_sample, df_grad_g_z, epoch_list, grid_size, quiver_name, animation_frame)
            fig_all = final_plot_quiver(fig_sample, fig_data, fig_quiver, frames_quiver, df_score_info, method=method_type, training=training)

    fig_loss = None
    if training:
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_dsc_itr, df_loss_gen_itr, df_score = read_hdf_loss(file)
        fig_loss = create_fig_loss(df_loss, df_loss_dsc_itr, df_loss_gen_itr, df_score)
        return fig_all, fig_loss
    else:
        return fig_sample, fig_loss

 


if __name__=='__main__':
    # file = r'saved_result\\2spirals\\saved_hdfs_training\\DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64_df_sample_per_epoch.h5'
    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    expr_id = 'DDPM_beta_linear_T_40_ToyDDPM_4_64_swissroll_t_dim_1'
    expr_id = 'FlowMatching_T_40_ToyFlowMatching_4_64_swissroll_t_dim_1_gamma_0.025'
    expr_id = 'Regression_ToyRegressionNet_4_64_swissroll'
    expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_500_gamma_0.025'
    expr_id = 'GAN-RKL_ToyGAN_4_64_2moons_z_dim_2_lr_dsc_1e-4_lr_gen_1e-4_loss_dsc_stan_lc_0.5_0.5_0.5_loss_gen_heur_lc_1'
    expr_id = 'GAN-wo-G_ToyGAN_4_64_swissroll_z_dim_3_lr_dsc_1e-04_lr_gen_1e-04_lr_fk_dt_1e-03_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    method = expr_id.split('_')[0]
    # fig_sample, fig_loss = plot_animation(method, expr_id, training=True, all_epochs=True, quiver_name='grad')
    fig_sample, fig_loss = plot_animation(method, expr_id, training=True, epoch_key=5, all_epochs=False, quiver_name='grad')
    # fig_sample, fig_loss = plot_animation_regression(method, expr_id, training=False, all_zeros=True)