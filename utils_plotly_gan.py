

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

def show_latex_in_plotly():
    from plotly import offline
    from IPython.display import display, HTML
    ## Tomas Mazak's workaround
    offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))
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
        df_score_info_hist = pd.DataFrame()
        # df_grad_g_z = pd.DataFrame()
        # df_grad_g_z_at_sampling = pd.DataFrame()
        df_fake_data = pd.DataFrame()
        # df_intermediate_smpl = pd.DataFrame()
        df_intermediate_smpl_list = []
        df_grad_g_z_list = []
        epoch_list = []
        for k in keys:
            if 'samples' in k: 
                df_sample = pd.concat([df_sample if not df_sample.empty else None, hdf_file[k]])
                epoch_list.append(int(k.split('/')[2].split('_')[2]))
            elif 'intermediate_smpl'in k:
                # df_intermediate_smpl = pd.concat([df_intermediate_smpl if not df_intermediate_smpl.empty else None, hdf_file[k]])
                df_intermediate_smpl_list.append(hdf_file[k])
            elif 'score_data_sample_info' in k:
                df_score_info = pd.concat([df_score_info if not df_score_info.empty else None, hdf_file[k]])
            elif 'score_data_sample_hist' in k: # score of samples generated iteratively
                df_score_info_hist = pd.concat([df_score_info_hist if not df_score_info_hist.empty else None, hdf_file[k]])
            # elif 'grad_g_z_at_sampling' in k:
            #     # df_grad_g_z_at_sampling = pd.concat([df_grad_g_z_at_sampling if not df_grad_g_z_at_sampling.empty else None, hdf_file[k]]) 
            elif 'grad_g_z' in k:
                # df_grad_g_z = pd.concat([df_grad_g_z if not df_grad_g_z.empty else None, hdf_file[k]])
                df_grad_g_z_list.append(hdf_file[k])
            elif 'fake_data' in k:
                df_fake_data = pd.concat([df_fake_data if not df_fake_data.empty else None, hdf_file[k]])

        df_data = hdf_file['/df/data']
        n_samples = hdf_file['/df/info']['n_samples'].values[0]
        grid_size = hdf_file['/df/info']['grid_size'].values[0]
        timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
        hdf_file.close()
    except:
        print(f'Error in reading hdf file {file}')
        hdf_file.close()

    return df_sample, df_data, df_intermediate_smpl_list, df_fake_data, df_score_info.reset_index(drop=True), \
            df_score_info_hist, df_grad_g_z_list, epoch_list, timesteps_list, n_samples, grid_size

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

def prepare_df_all_epochs_for_plotly(df_sample, df_grad_g_z, df_score_info, epoch_list, n_samples, grid_size):

    df_sample['epoch'] = np.repeat(epoch_list, n_samples)
    # df_sample['type'] = [f'sample_{i}' for i in range(1, n_samples+1)] * len(epoch_list)
    # if df_data.shape[0] > 2000:
    #     df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    # n_data = df_data.shape[0]
    # df_data = pd.concat([df_data] * len(epoch_list)).reset_index(drop=True)
    # df_data['epoch'] = np.repeat(epoch_list, n_data)
    # df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    # n_data = df_fake_sample.shape[0]
    # df_fake_sample = pd.concat([df_fake_sample] * len(epoch_list)).reset_index(drop=True)
    # df_fake_sample['epoch'] = np.repeat(epoch_list, n_data)
    # df_fake_sample['type'] = [f'grid_sample_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    df_grad_g_z['epoch'] = np.repeat(epoch_list, grid_size**2)
    df_score_info['epoch'] = epoch_list

    animation_frame = 'epoch'
    return df_sample, df_grad_g_z, df_score_info, animation_frame

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




def add_all_zeros_animation(df_sample, df_grad_g_z_list, df_grid_samples, epoch_list, colors, quiver_name='grad'):
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    
    df_sample = df_sample[::-1] # last sample will be at the end of animation
    px_anim_sample = px.scatter(df_sample, x='x', y='y', animation_frame='epoch')
    xy = df_grid_samples
    x, y = xy['x'].to_numpy(),  xy['y'].to_numpy() # inputs of ff.create_quiver must be numpy or pandas not tensor to function faster
    quiver_data = []
    for dfg in df_grad_g_z_list:
        u, v = dfg['u'],  dfg['v']
        # u, v = u.reshape(-1, grid_size), v.reshape(-1, grid_size)
        scale = np.sqrt(np.linalg.norm(x) / np.linalg.norm(u)) + np.sqrt(np.linalg.norm(y) / np.linalg.norm(v))
                    # u is added to x and v is added to y.
        fig_quiver = ff.create_quiver(x, y, u, v, scale=scale, name=f'{quiver_name}', arrow_scale=.2,
                                                angle=40*np.pi/180, line=dict(width=1, color=colors['grad']))
        quiver_data.append([fig_quiver.data[0]])

    all_zero_sample_frames = [go.Frame(data=[go.Scatter(x=smpl_frame.data[0].x, y=smpl_frame.data[0].y,  mode='markers', showlegend=True, 
                                             line=dict(color=colors['sample']), name=f'sample', hovertemplate='x=%{x}<br>y=%{y}'),
                                             ] + grad_frame,  
                            traces=[1, 2], name=f'zero_epoch_{e}', group=f'all_zero')  
                            for e, smpl_frame, grad_frame in zip(epoch_list, px_anim_sample.frames, quiver_data)]

    return all_zero_sample_frames 

def add_epochs_animation_slider(df_intermediate_smpl_list, epoch_list, timesteps_list, colors):
    # add animations of all epochs
    animation_list = []
    for i, (e, dfs) in enumerate(zip(epoch_list, df_intermediate_smpl_list)): #traces: data samples grads

        dfs = dfs[::-1] # last sample will be at the end of animation
        px_anime_sample = px.scatter(dfs, x='x', y='y', animation_frame='time')
        sample_frames = [go.Frame(data=[go.Scatter(x=frame['data'][0].x, y=frame['data'][0].y,  mode='markers', showlegend=True,
                                                line=dict(color=colors['sample']), name=f'sample_{e}', hovertemplate='x=%{x}<br>y=%{y}')
                                     ], traces=[3+i], name=f'frame_{e}_{t}', group=f'epoch_{e}')  
                                     for t, frame in zip(timesteps_list[::-1], px_anime_sample.frames)]

        animation_list.append(sample_frames)

    return animation_list

def add_data_sammple_trace(fig, df_data, all_zero_sample_frames, animation_list, epoch_list, colors):

    # data, sample_zero at first epoch , grad at first epoch
    fig.add_trace(go.Scatter(x=df_data['data_x'], y=df_data['data_y'], mode='markers', line=dict(color=colors['data']), name=f'data',
                            visible=True, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    fig.add_trace(go.Scatter(x=all_zero_sample_frames[0].data[0].x, y=all_zero_sample_frames[0].data[0].y, mode='markers', 
                             line=dict(color=colors['sample']), name=f'sample', visible=True, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    fig.add_trace(go.Scatter(x=all_zero_sample_frames[0].data[1].x, y=all_zero_sample_frames[0].data[1].y, mode='lines', 
                             line=dict(color=colors['grad']), name=f'grad', visible=True, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    
    # sample trace of all epoch in epoch_list
    for e, anim in zip(epoch_list, animation_list):
        fig.add_trace(go.Scatter(x=anim[0].data[0].x, y=anim[0].data[0].y, mode='markers', 
                                line=dict(color=colors['sample']), name=f'sample_{e}', visible=False, showlegend=True))
    return fig

def add_statistics_to_fig(fig, df_loss_dsc_itr, df_loss_gen_itr, df_loss, df_score):

    # add traces of statistics
  
    x = df_loss_dsc_itr['itr']
    fig.add_trace(go.Scatter(x=x, y=df_loss_dsc_itr['loss_dsc_itr'], name="Loss D", mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen')))
    fig.add_trace(go.Scatter(x=x, y=df_loss_gen_itr['loss_gen_itr'], name="Loss G", mode='lines', visible=False, showlegend=True, line=dict(width=1, color='crimson'))) 
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss_dsc_itr['loss_dsc_itr']) / x), name="Loss D mean", mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss_gen_itr['loss_gen_itr']) / x), name="Loss G mean", mode='lines', visible=False, showlegend=True, line=dict(width=2, color='deeppink')))
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

    # plot std of scores in epochs, not std of mean of scores
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
    
    
    return fig

def visibility_traces(fig):
    # visibility of traces when coresponding button is clicked
    trace_visibility = dict()
    for trace in fig.data:
        if trace.name not in  trace_visibility.keys():
            trace_visibility[trace.name] = False
        elif 'Loss' in trace.name:
            trace_visibility[f'{trace.name} epoch'] = False

    # trace_names_example = ['data', 'sample', 'grad', 
    #                'sample_10', 'sample_20', 'sample_30', 'sample_40', 'sample_50', 'sample_60', 'sample_70', 'sample_80', 'sample_90', 'sample_100', 
    #                'Loss D', 'Loss G', 'Loss D mean', 'Loss G mean', 
    #                'Loss D', 'Loss G', 'Loss D mean', 'Loss G mean', 
    #                'D(x)', 'D(G(z))', 
    #                'D(x) std', 'D(G(z)) std', 'D(x) mean', 'D(G(z)) mean', 'D*']
    
    trace_visibility['data'] = True
    

    trace_vis_loss_itr = trace_visibility.copy()
    trace_vis_loss_itr.update({f'Loss D': True, f'Loss G': True, f'Loss D mean': True, f'Loss G mean': True, 'data': False})
    trace_vis_loss_epoch = trace_visibility.copy()
    trace_vis_loss_epoch.update({f'Loss D epoch': True, f'Loss G epoch': True, f'Loss D mean epoch': True, f'Loss G mean epoch': True, 'data': False})
    trace_vis_score_epoch = trace_visibility.copy()
    trace_vis_score_epoch.update({f'D(x)': True, f'D(G(z))': True, f'D*': True, 'data': False})
    trace_vis_score_mean = trace_visibility.copy()
    trace_vis_score_mean.update({f'D(x) std': True, f'D(G(z)) std': True, f'D(x) mean': True, f'D(G(z)) mean': True, f'D*': True, 'data': False})
    
    stats_trace_vis = [trace_vis_loss_itr, trace_vis_loss_epoch, trace_vis_score_epoch, trace_vis_score_mean]

    return trace_visibility, stats_trace_vis

def visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics):

    # store idx of other buttons that adding to the layout for making visible True or False
    stats_idx = n_button_animations
    pause_idx = n_button_animations + n_button_statistics
    back_stats_idx = pause_idx + 1
    back_detail_idx = back_stats_idx + 1
    select_dropdown_idx = back_detail_idx + 1 
     
    # buttons visibility
    button_visibility = {}
    for i in range(n_all_buttons):
        button_visibility[f'updatemenus[{i}].visible'] = False 
    button_visibility[f'updatemenus[{pause_idx}].visible'] = True
    button_visibility[f'updatemenus[{select_dropdown_idx}].visible'] = True

    return button_visibility, stats_idx, pause_idx, back_stats_idx, back_detail_idx, select_dropdown_idx

def add_play_show_buttons_slider(trace_visibility, button_visibility, timesteps_list, button_info, slider_info, epoch_list):
    
    play_button_list = []
    select_button_list = [] #drop down button

    # Play buttons for sampling
    frame_spec = {'frame': {'duration': 300, 'redraw': False}, 'fromcurrent': True, 'mode': 'immediate', 
              'transition': {'duration': 20, 'easing': 'cubic-in-out'}}
    

    zero_play_button = dict(buttons=[dict(label=f"{f'&#9654; zeros':^16}", method='animate', args=[f'all_zero' , frame_spec])])
    zero_play_button.update(button_info)
    zero_play_button.update(dict(x=-0.22, y=0.001))

    play_button_list.append(zero_play_button)
    # slider for zero play button
    
    all_zero_slider = dict(steps=[dict(args=[[f'zero_epoch_{e}'], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate' }], 
                                method='animate', label=f'{e}' ) for e in epoch_list] )
    all_zero_slider.update(slider_info)

    visibility_layout = {'title': f'All zeros'}
    visibility_layout.update(button_visibility)
    visibility_layout[f'updatemenus[0].visible'] = True # show play zeros buttons
    visibility_layout['sliders'] = [all_zero_slider] 

    # add fist button to dropdwon 
    trace_vis_init = trace_visibility.copy()
    trace_vis_init.update({f'sample': True, f'grad': True})
    select_button_list.append(dict(label=f'Show all zeros', method='update', 
                                args=[{'visible': list(trace_vis_init.values())}, visibility_layout]))

    vis_layout = visibility_layout.copy()
    vis_layout[f'updatemenus[0].visible'] = False
   # add play buttons for all epochs
    for i, e in enumerate(epoch_list):

        slider_steps = dict(steps=[dict(args=[[f'frame_{e}_{t}'], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate' }], 
                                    method='animate', label=f'{t}' ) for t in timesteps_list[::-1]])
        slider_steps.update(slider_info)
        slider_steps.update({'currentvalue': dict(font_size=12, prefix= f'Epoch{e}: ', xanchor='left') , 'name': f'e_{e}'})
        
        play_button = dict(buttons=[dict(label=f"{f'&#9654; {e}':^20}", method='animate', args=[f'epoch_{e}', frame_spec])])
        play_button.update(button_info)
        play_button.update(dict(visible=False, x=-0.22, y=0.001))
        play_button_list.append(play_button)
        vis = {}
        vis.update(vis_layout)
        vis['title'] =  f'Epoch {e}'
        vis['sliders'] = [slider_steps]
        vis[f'updatemenus[{i+1}].visible'] = True # first play button is for all zeros
        
        trace_vis_epoch = trace_visibility.copy()
        trace_vis_epoch.update({f'sample_{e}': True})

        select_button_list.append(dict(label=f'Show epoch {e}', method='update', 
                                       args=[{'visible': list(trace_vis_epoch.values())}, vis]))  
        

    select_button_list.append(dict(label=f'', method='update',  args=[None]))

    return play_button_list, select_button_list, all_zero_slider

def add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, n_button_animations, button_info, method,
                      back_stats_idx, pause_idx, select_dropdown_idx):

    # Show all stats buttons (loss, score)
    stats_button_list = []
    show_stats = {'title': method, 'width':1000, 'height': 450, 'xaxis.autorange': True, 'yaxis.autorange': True} 
                  #'xaxis.range': [-2, 9], 'yaxis.range': [-3, 7]
    show_stats['sliders'] = []
    show_stats.update(button_visibility) 
    show_stats[f'updatemenus[{back_stats_idx}].visible'] = True
    show_stats[f'updatemenus[{pause_idx}].visible'] = False
    show_stats[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    
    stats_lables = ['Loss itr', 'Loss epoch', 'Score', 'Score mean']
    for i in range(n_button_statistics):  
        j = i + n_button_animations
        show_stats[f'updatemenus[{j}].visible'] = True 

        stats_vis = stats_trace_vis[i]
        stats_button = dict(buttons=[dict(label=f'{stats_lables[i]}', method='update', args=[{'visible': list(stats_vis.values())}, 
                                                                                            {'title': f'{stats_lables[i]}'}])])
        stats_button.update(button_info)
        stats_button.update(dict(visible=False, x=0.15+((i+1))/8, y=1.02, pad={"r": 10, "t": 25, "l": 10}))
        stats_button_list.append(stats_button)
    # Show loss in select button [dropdown]
    stats_vis = stats_trace_vis[0]
    show_stats_button = dict(label=f'Show Loss-Score', method='update', args=[{'visible': list(stats_vis.values())}, show_stats])

    return stats_button_list, show_stats_button
    
def add_detail_button(trace_visibility, button_visibility, config, back_detail_idx, pause_idx, select_dropdown_idx):

    # show details  in select button [dropdown ]
    show_detail = {}
    show_detail.update(button_visibility)
    show_detail[f'updatemenus[{back_detail_idx}].visible'] = True
    show_detail[f'updatemenus[{pause_idx}].visible'] = False
    show_detail[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    # Formatted string with LaTeX

    title_text = r"$\begin{aligned}"
    for k, v in config.items():
        if 'save_dir' in k:
            continue
        key = k.replace("_", "-")
        value = str(v).replace("_", "-")
        if  value.isdigit():
            title_text += rf" & \text{{{key}}}  \quad & = \quad & {value}  \\ "
        else:
            title_text += rf" & \text{{{key}}}  \quad & = \quad & \text{{{value}}}  \\ "
    title_text += r"\end{aligned}$"  # End LaTeX block
    show_detail['title.text'] = title_text
    show_detail['sliders'] = []
    
    show_detail.update({'margin.t':100, 'title.y': 0.6, 'title.x':0.2, 'xaxis.range': [-2.5, 2.5], 'yaxis.range': [-2.5, 2.5], 
                            'xaxis.tickvals': [], 'yaxis.tickvals': [], 'xaxis.title': '', 'yaxis.title': ''
                            }) 
    show_detail.update(dict(plot_bgcolor='white', height=450))
    hide_traces = trace_visibility.copy()
    hide_traces['data'] = False
    show_detail_button = dict(label=f'Show details', method='update', args=[{'visible': list(hide_traces.values())}, show_detail])
            #    [dict(label=f'Hide Loss-Score', method='update', args=[{'y': [], 'x': []}, hide_loss]),
            #    dict(label=f'1 details', method='relayout', args=[{'title': f'GAN 1'}])]
    return show_detail_button

def add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, method,
                     back_detail_idx, stats_idx, select_dropdown_idx, back_stats_idx):

    hide_info = {'margin.t': 60, 'title.y': 0.98, 'title.x':0.001, 'showlegend': True, 'plot_bgcolor': None, 'width':750, 'height': 570,
                'xaxis.tickvals': None, 'yaxis.tickvals': None, 'xaxis.range': [-2.5, 2.5], 'yaxis.range': [-2.5, 2.5], 
                'xaxis.autorange': False, 'yaxis.autorange': False, 'xaxis.title': 'x', 'yaxis.title': 'y'}

    slider_steps = dict(steps=[dict(args=[[None]], method='animate', label=f'{j}' ) for j in range(2)])
    slider_steps.update(slider_info)
    slider_steps.update({'currentvalue': dict(font_size=12, prefix= f'', xanchor='left')})
    hide_info['sliders'] = [slider_steps]

    # add back button [stats]
    hide_stats = {'title': method}
    hide_stats.update(hide_info)
    # changing vsibility to False must be after changing title and slider to slider work properly
    for i in range(n_button_statistics):
        hide_stats[f'updatemenus[{stats_idx+i}].visible'] = False

    
    hide_stats[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_stats[f'updatemenus[{back_stats_idx}].visible'] = False
    back_stats_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_visibility.values())}, hide_stats])]) 
    back_stats_button.update(button_info)
    back_stats_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))

    # add back button  [details]
    hide_detail = {'title.text': method}
    hide_detail.update(hide_info) 
    hide_detail[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_detail[f'updatemenus[{back_detail_idx}].visible'] = False
    back_detail_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_visibility.values())}, hide_detail])]) 
    back_detail_button.update(button_info)
    back_detail_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))
    
    return back_stats_button, back_detail_button
    




def prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, df_grid_samples,
                                df_grad_g_z_list, df_score_info, df_score_info_hist,
                                df_loss_dsc_itr, df_loss_gen_itr, df_loss, df_score,
                                epoch_list, timesteps_list, quiver_name, method, config,
                                width=750, height=570, range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]):

    # The traces must be added in the correct order for the button's visibility to function properly
   
    num_epochs = len(epoch_list)
    colors = dict(sample='#AB63FA', grad='DarkSlateGrey', data='#19D3F3') # samples-grads-data
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    all_zero_sample_frames = add_all_zeros_animation(df_sample, df_grad_g_z_list, df_grid_samples, epoch_list, colors, quiver_name)

    # add animations of all epochs
    animation_list = add_epochs_animation_slider(df_intermediate_smpl_list, epoch_list, timesteps_list, colors)

    fig = go.Figure() 
    # data, sample_zero at first epoch , grad at first epoch
    fig = add_data_sammple_trace(fig, df_data, all_zero_sample_frames, animation_list, epoch_list, colors)

    # add traces of statistics
    fig = add_statistics_to_fig(fig, df_loss_dsc_itr, df_loss_gen_itr, df_loss, df_score)

    # visibility of traces when coresponding button is clicked
    trace_visibility, stats_trace_vis = visibility_traces(fig)

    n_button_animations = len(animation_list) + 1  # 1: for all_zeros button # equals to plays buttons
    n_button_statistics = 4 # loss itr, loss epoch, score epoch, score mean & std
    n_all_buttons = n_button_animations + n_button_statistics + 2 + 2  # show stats, show detail, show eval, 
                                                                            # back_stats, back_detail, back_eval
    # n_traces_animation = 1 + 1 +  num_epochs # 1:data, 1: grad first epoch, len(epoch_list): samples of each epoch
    # n_traces_statistics = 4 + 4 + 2 + 4 + 1 # 4: loss itr 4: loss epoch, 2: score epoch, 4: score mean & std, 1 is for D*

    # store idx of other buttons that adding to the layout for making visible True or False
    # buttons visibility
    button_visibility, stats_idx, pause_idx, \
    back_stats_idx, back_detail_idx, select_dropdown_idx = visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics)

    button_info = dict(visible=True, type='buttons', direction='down', showactive=False, pad={"r": 10, "t": 10, "l": 10}, 
                         xanchor="right", yanchor="bottom", bgcolor='seashell')
    slider_info = dict(active=0, name=f'all_epoch', visible=True, currentvalue=dict(font_size=12, prefix=f'zero of epoch: ', xanchor='left'),
                 transition=dict(duration=300, easing='cubic-in-out'), pad=dict(b=10, t=60), len=1, x=0.0, y=0.08, xanchor='left', yanchor='top')
    # add play buttons, show buttons
    play_button_list, select_button_list, all_zero_slider = add_play_show_buttons_slider(trace_visibility, button_visibility, timesteps_list, 
                                                                                         button_info, slider_info, epoch_list)
    # show all stats buttons (loss, score)
    stats_button_list, show_stats_button = add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, 
                                                             n_button_animations, button_info, method,
                                                                back_stats_idx, pause_idx, select_dropdown_idx)


    # show details  in select button [dropdown ]
    show_detail_button = add_detail_button(trace_visibility, button_visibility, config, back_detail_idx, pause_idx, select_dropdown_idx)
    
    # add pause button
    pause_button = dict(buttons=[dict(label=f"{f'&#9724;':^5}", method='animate',  args=[[None],  {'frame':{'duration': 0, 'redraw': False}, 
                                            'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'linear'}}])])
    pause_button.update(button_info)
    pause_button.update(dict(x=-0.12, y=0.001))

    # [ back buttons ]
    back_stats_button, back_detail_button = add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, 
                                                        method, back_detail_idx, stats_idx, select_dropdown_idx, back_stats_idx)
    
    button_list = play_button_list + stats_button_list + [pause_button] + [back_stats_button] + [back_detail_button]

    # all buttons          
    updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_detail_button]+[show_stats_button]+select_button_list[1:],
                                        visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
                                        x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')]
    # print(len(updatemenues))  
    fig.layout=go.Layout(
            updatemenus=updatemenues,
            sliders=[all_zero_slider])
    frames = all_zero_sample_frames
    for i in range(len(animation_list)):
        frames += animation_list[i]
    fig.frames = frames
    # Update layout for better aesthetics
    # scaleanchor='y', scaleratio=1,
    # scaleanchor='x', scaleratio=1,
    fig.update_layout(
        title=dict(text=method, pad={"r": 10, "t": 10, "l": 20, "b":10}, x=0.001, xanchor="left", y=0.98, yanchor="top"),
        xaxis=dict(title=dict(text="x", font_size=12), range=range_x,  showgrid=True, autorange=False),
        yaxis=dict(title=dict(text="y", font_size=12), range=range_y,  showgrid=True, autorange=False),
        # plot_bgcolor='rgba(100, 100, 100, 0)',
        margin=dict(l=80, t=60, b=20, r=150),
        showlegend=True,
        width=width,
        height=height
    )
    return fig

 

def plot_animation(method, expr_id, training=True, epoch_key=-1, all_epochs=False, quiver_name=None, test_name=None):

    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_sample_per_epoch.h5' if training else f'{path}/{expr_id}_df_sample.h5'  
    method_type = f"{method} [{config['loss_dsc']} {config['loss_gen']}]"
    
    if test_name is not None:
       file = f'{path}/{expr_id}_df_sample_{test_name}.h5' 


    if training and all_epochs:
        df_sample, df_data, df_intermediate_smpl_list, df_fake_data, df_score_info, df_score_info_hist, df_grad_g_z_list,\
              epoch_list, timesteps_list, n_samples, grid_size = read_hdf_all_epochs_train(file)

        grid_samples, grid_x, grid_y = get_fake_sample_grid(grid_size) # for ploting quiver
        df_grid_samples = pd.DataFrame(grid_samples.numpy(), columns=['x', 'y'])
        # get statistics (loss, score) df
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_dsc_itr, df_loss_gen_itr, df_score = read_hdf_loss(file)

        fig_all = prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, df_grid_samples, 
                                                            df_grad_g_z_list, df_score_info, df_score_info_hist,
                                                            df_loss_dsc_itr, df_loss_gen_itr, df_loss, df_score,
                                                            epoch_list, timesteps_list, quiver_name, method_type, config)
          
    show_latex_in_plotly()
    return fig_all


 


if __name__=='__main__':
    # file = r'saved_result\\2spirals\\saved_hdfs_training\\DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64_df_sample_per_epoch.h5'
    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    expr_id = 'DDPM_beta_linear_T_40_ToyDDPM_4_64_swissroll_t_dim_1'
    expr_id = 'FlowMatching_T_40_ToyFlowMatching_4_64_swissroll_t_dim_1_gamma_0.025'
    expr_id = 'Regression_ToyRegressionNet_4_64_swissroll'
    expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_500_gamma_0.025'
    expr_id = 'GAN-RKL_ToyGAN_4_64_swissroll_z_dim_1_lr_dsc_1e-4_lr_gen_1e-4_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    expr_id = 'GAN-wo-G_ToyGAN_4_64_swissroll_z_dim_3_lr_dsc_1e-04_lr_gen_1e-04_lr_fk_dt_1e-03_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    method = expr_id.split('_')[0]
    # fig_sample, fig_loss = plot_animation(method, expr_id, training=True, all_epochs=True, quiver_name='grad')
    fig_sample = plot_animation(method, expr_id, training=True, epoch_key=5, all_epochs=True, quiver_name='grad')
    # fig_sample, fig_loss = plot_animation_regression(method, expr_id, training=False, all_zeros=True)