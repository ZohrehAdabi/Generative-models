


import argparse
# from  json import loads as list_type

def parse_args(method='Diffusion', script='sampling'):

    parser = argparse.ArgumentParser(description=f'{method} {script}')

    parser.add_argument('--seed'                , default=0, type=int, help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--model'               , default='ToyNetwork2', help='model:ToyNetwork{1|2|3|4|5|6} / ') 
    parser.add_argument('--data_dim'            , default=2, type=int, help ='Data dimension, used in creation of model')

    parser.add_argument('--dataset'             , default='2spirals', help='25gaussians/8gaussians/swissroll/2spirrals/2circles/2sines/checkerboard/2moons')
    parser.add_argument('--batch_size'          , default=128, type=int, help ='For creating DataLoader')
    parser.add_argument('--total_size'          , default=20000, type=int, help ='Number of datapoints in generating synthesized dataset')
    parser.add_argument('--normalize'           , action='store_true', help='Normalize dataset, (x-mu)/std| minmax scalar')

    parser.add_argument('--n_epoch'             , default=5, type=int, help ='Use this when do not determine start and stop epoch') 
    parser.add_argument('--lr'                  , default=1e-3, type=float, help ='Learning rate for decoder/generator')
    
    parser.add_argument('--n_samples'           , default=500, type=int, help='Number of samples for generation') 

    if method in ['DDPM', 'FlowMatching', 'Boosting', 'BoostingOne', 'Regression']: 
        parser.add_argument('--beta_schedule'       , default='linear', help='{linear|quadratic|sigmoid|cosine}')
        parser.add_argument('--n_timesteps'         , default=10, type=int, help ='Number of time steps for noise scheduling (sampling)')
        parser.add_argument('--time_dim'            , default=16, type=int, help ='Used in setting dimension of out time embedding (sinusoidal positional embedding)')
        parser.add_argument('--n_timestep_smpl'     , default=-1, type=int, help ='Number of time steps for sampling')
        parser.add_argument('--n_sel_time'          , default=10, type=int, help='Number of time selected to plot fewer plot of samples in animation, n_timesteps%n_sel_time==0') 
        parser.add_argument('--grid_size'           , default=32, type=int, help ='Grid size for generating and ploting gradient in a grid')
        
    if method in ['GAN-RKL', 'GAN-wo-G']: 
        parser.add_argument('--z_dim'               , default=8, type=int, help ='Dimension of z input for generator')
        parser.add_argument('--lr_dsc'              , default=1e-3, type=float, help ='Learning rate for encoder or discriminator')
        parser.add_argument('--lr_gen'              , default=1e-3, type=float, help ='Learning rate for decoder/generator')
        parser.add_argument('--lr_fk_dt'            , default=1e-3, type=float, help ='Learning rate for fake_data optimization')
        parser.add_argument('--loss_dsc'            , default='stan',  help ='loss function for training Discriminator of GAN [stan|heur|comb|rvrs]')
        parser.add_argument('--loss_gen'            , default='huer',  help ='loss function for training Generator of GAN [stan|heur|comb|rvrs]')
        parser.add_argument('--grid_size'           , default=32, type=int, help ='Grid size for generating and ploting gradient in a grid')

        parser.add_argument('--n_timestep_smpl'     , default=40, type=int, help ='Number of time steps for sampling')
        parser.add_argument('--lc_dsc'              , default="", type=str, help ='Coefficients used in discreiminator loss')
        parser.add_argument('--lc_gen'              , default="", type=str, help ='Coefficients used in generator loss')
        parser.add_argument('--n_sel_time'          , default=10, type=int, help='Number of time selected to plot fewer plot of samples in animation, n_timesteps%n_sel_time==0') 
    
    if method in ['BoostingOne']: 
        parser.add_argument('--pred_goal'           , default='grad', help='Determining using grad or noise in prediction')
        parser.add_argument('--gamma'               , default=0.1, type=float, help ='coefficient of gradient in x = x + gamma * grad')
    
    if method in ['Boosting']:    
        parser.add_argument('--learner_inp'         , default='pred_x', help='Determining using x or noise or predicted_x or x_noisy for weak learner in Boosting')
        parser.add_argument('--innr_epoch'          , default=5, type=int, help ='Number of epoch for training model at time t')
        parser.add_argument('--grad_type'           , default='pred_x', help="Determining grad type (noise - x) or (pred_x - x)")
        parser.add_argument('--gamma'               , default=0.1, type=float, help ='coefficient of gradient in x = x + gamma * grad')

    
    if script == 'train':
        
        parser.add_argument('--start_epoch'     , default=0, type=int, help ='Starting epoch')
        parser.add_argument('--stop_epoch'      , default=-1, type=int, help ='Stopping epoch')
        parser.add_argument('--resume'          , action='store_true', help='Continue from previous trained model with largest epoch')
        
        parser.add_argument('--save_model'      , action='store_true', help='Save model in some epochs.')
        parser.add_argument('--save_freq'       , default=50, type=int, help='Save frequency during training')
        parser.add_argument('--validation'      , action='store_true', help='Run sampling of Diffusion')
        parser.add_argument('--save_hdf'        , action='store_true', help='Save statistics of training in hdf file')
        parser.add_argument('--save_fig'        , action='store_true', help='Save figures of generated samples in validation')
        parser.add_argument('--save_eval'       , action='store_false', help ='Save evaluation metrics {IS, FID, Mode score, RKL, Missing Modes}')
    
    elif script == 'sampling':
        # parser.add_argument('--n_samples'       , default=500, type=int, help='Number of samples for generation') 
        parser.add_argument('--validation'      , action='store_false', help='Run sampling of Diffusion')
        parser.add_argument('--save_model'      , action='store_false', help='Save model in some epochs.')
        parser.add_argument('--save_eval'       , action='store_true', help ='Save evaluation metrics {IS, FID, Mode score, RKL, Missing Modes}')
        parser.add_argument('--save_fig'        , action='store_true', help='Save figures of generated samples')
        parser.add_argument('--save_hdf'        , action='store_true', help='Save hdf file of generated samples')


    return parser.parse_args()
