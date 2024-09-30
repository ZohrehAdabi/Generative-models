from common_imports_diffusion import *

from scipy.integrate import solve_ivp



class BoostingOne_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None, 
            betas=None, 
            n_timestep_smpl=100, 
            expr_id=None, 
            training=False
        ):
        super(BoostingOne_Sampler, self).__init__()

        self.n_timestep_smpl= n_timestep_smpl
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
        
 
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 
    

    def sampling(self, pred_goal, n_samples=5, gamma=0.001, data_dim=2, solver='euler', 
                 normalize=False, scaler=None, params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()

        

        sample = torch.randn(n_samples, data_dim).to(device)
        sample_noise = torch.randn(n_samples, data_dim).to(device)
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, data_dim])
        intermediate_smpl[-1] = sample.cpu()
        grad_or_noises = np.empty([self.n_timestep_smpl, n_samples, data_dim])
        grad_or_noise_norm_list = []

        if pred_goal=='noise':
            selected_t = torch.tensor([1])
            timestp = tqdm(timesteps, disable=self.training)
            for t in timestp: 
                
                times = torch.repeat_interleave(selected_t, n_samples).reshape(-1, 1).long().to(device)
                with torch.no_grad():
                    sample, predicted_noise = self.sampling_ddpm_step(sample, times)
                    grad_or_noise_norm_list.append([torch.linalg.norm(predicted_noise, axis=1).mean().cpu().numpy()])
                    grad_or_noises[t, :, :] = predicted_noise.cpu().numpy()
                if normalize:
                    intermediate_smpl[t, :, :] = scaler.inverse_transform(sample.cpu())
                else:
                    intermediate_smpl[t, :, :] = sample.cpu()
            timestp.close()  
            sample_zero = intermediate_smpl[0]

        else: #pred_goal=='grad'

            if solver=='euler':
                timestp = tqdm(timesteps, disable=self.training)
                for t in timestp: 
                    # times = torch.repeat_interleave(t, n_samples).reshape(-1, 1).long().to(device)
                    with torch.no_grad():
                        predicted_grad = self.sampling_boosting_step_euler(sample)
                        sample = sample + 0.025 * predicted_grad
                        grad_or_noise_norm_list.append([torch.linalg.norm(predicted_grad, axis=1).mean().cpu().numpy()])
                        grad_or_noises[t, :, :] = predicted_grad.cpu().numpy()
                    if normalize:
                        intermediate_smpl[t, :, :] = scaler.inverse_transform(sample.cpu())
                    else:
                        intermediate_smpl[t, :, :] = sample.cpu()
                timestp.close()  
                sample_zero = intermediate_smpl[0]

            elif solver=='ode':
            
                with torch.no_grad():
                    sample = self.sampling_boosting_step_ode(sample.cpu()).reshape([-1, n_samples, data_dim])
                    # sample = sample + gamma * predicted_grad
                if normalize:
                        intermediate_smpl[:-1, :, :] = scaler.inverse_transform(sample)
                else:
                    intermediate_smpl[:-1, :, :] = sample
                    # intermediate_smpl[t, :, :] = sample.cpu()
                sample_zero = intermediate_smpl[0]

        grad_or_noise_norm = np.concatenate(grad_or_noise_norm_list) 
        self.save_result(sample_zero, intermediate_smpl, grad_or_noise_norm, grad_or_noises, params)

        return sample_zero, intermediate_smpl#[:-1]

    def sampling_boosting_step_euler(self, x):

        pred_grad = self.model(x)
        return pred_grad

    def sampling_boosting_step_ode(self, x):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        solution = solve_ivp(self.ode_func, (1, 0.001), x.reshape([-1,]), args=(self.n_timestep_smpl,),
                                                rtol=1e-5, atol=1e-5, method='RK45', dense_output=True)
        # nfe = solution.nfev
        t = np.linspace(0, 1, self.n_timestep_smpl)
        bacward_ode = solution.sol(t).T
        
        return bacward_ode
    
    def ode_func(self, t, x, N):
            
        time = int(t * (N - 1) / 1)
        inp = torch.tensor(x.reshape([-1, 2])).to(torch.float).to(self.model.device)
        predicted_grad = - self.model(inp).cpu().numpy().reshape([-1,])
        
        return predicted_grad
    
    def sampling_ddpm_step(self, x, t):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        predicted_noise = self.model(x)

        mu = self.compute_mu_theta(x, t, predicted_noise)
        sigma2 = torch.tensor([0.0006]).to(x.device) #self.reverse_variance(t)
        x = mu + torch.sqrt(sigma2) * torch.randn_like(x) * int((t>0).all())

        return x, predicted_noise

    def compute_mu_theta(self, x, t, predicted_noise):
        """
        DDPM  
          approximated posterior of forward process 
            mu = 1/(1-beta_t) (x_t - beta_t/sqrt(1 - alpha_t) * eps_theta)
        """
        # b = torch.tensor([0.0006]).to(x.device)#self.betas[t]
        # s_one_alpha = torch.tensor([0.266]).to(x.device)# self.sqrt_one_minus_alphas[t]
        b = self.betas[t]
        s_one_alpha = self.sqrt_one_minus_alphas[t]
        return (1/torch.sqrt(1-b)) * (x - (b/s_one_alpha) * predicted_noise)
    
    def reverse_variance(self, t):
        """
        DDPM
        No training for Sigma [matrix].
        sigma_t = beta_t [scalar]  or
        sigma_t = (1- alpha_{t-1})/(1 - alpha_t) * beta_t [approximated posterior of forward process variance]
        """ 
        rev_variance = self.betas[t]
        # rev_variance = rev_variance.clip(1e-20)
        return rev_variance 
    
    def save_result(self, sample, intermediate_smpl, grad_or_noise_norm, grad_or_noises, params):

        if not self.training:
            create_save_dir(params.save_dir)
            if params.save_fig:
                f_name = f'sample_{self.expr_id}' if params.test_name is None else f'sample_{self.expr_id}_{params.test_name}'
                plot_samples(sample, f'{params.save_dir}/plot_samples/{f_name}.png')
                save_animation(intermediate_smpl, f'{params.save_dir}/plot_samples/{f_name}.mp4')
            if params.save_hdf:
                n_sel_time = params.n_sel_time if params.n_sel_time <= self.n_timestep_smpl else self.n_timestep_smpl 
                file_sample = create_hdf_file(params.save_dir, self.expr_id, params.dataset, 
                                              self.n_timestep_smpl, n_sel_time, params.n_samples, params.test_name)

                import warnings
                warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                # step = self.n_timestep_smpl // params.n_sel_time
                # dfs = pd.DataFrame(intermediate_smpl[::step, :, :].reshape(-1, data_dim), columns=['x', 'y'])
                intermediate_smpls = select_samples_for_plot(intermediate_smpl, self.n_timestep_smpl, n_sel_time)
                dfs = pd.DataFrame(intermediate_smpls, columns=['x', 'y'])
       

                grad_or_noises = select_samples_for_plot(grad_or_noises, self.n_timestep_smpl, n_sel_time)
                
                dfg = pd.DataFrame(grad_or_noises, columns=['x', 'y'])
                dfgi = pd.DataFrame({'time': np.arange(len(grad_or_noise_norm)-1, -1, -1), 'norm': grad_or_noise_norm})

                with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                    hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 
                    hdf_store_samples.append(key=f'df/grads', value=dfg, format='t')
                    hdf_store_samples.append(key=f'df/grad_info', value=dfgi, format='t')


if __name__=='__main__':

    method = 'BoostingOne'

    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)

    beta_schedule = params.beta_schedule
    n_timesteps = params.n_timesteps

    model_name = params.model
    data_dim= params.data_dim
    time_dim= params.time_dim
    hidden_dim= params.hidden_dim

    dataset_name = params.dataset
    n_samples = params.n_samples


    pred_goal = params.pred_goal
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl

    n_epochs = params.n_epoch
    lr = params.lr 
    inner_epoch = params.innr_epoch
    gamma = params.gamma
    params.test_name = None
    experiment_info = [
        ['method:', method],
        ['beta_schedule:', beta_schedule],
        ['n_timesteps:', n_timesteps],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['time_dim:', time_dim],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['total_size:', params.total_size],
        ['normalize:', params.normalize],
        ['pred_goal:', pred_goal],
        ['n_timestep_smpl:', n_timestep_smpl],
        ['n_epochs:', n_epochs],
        ['innr_epoch:', inner_epoch],
        ['lr:' , lr],
        ['gamma:', gamma],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')

    expr_id = f'BoostingOne_T_{n_timesteps}_{model_name}_{dataset_name}_pred_goal_{pred_goal}_gamma_{gamma}' 
                    #  f'_innr_ep_{inner_epoch}_gamma_{gamma}'
    if params.normalize:
        expr_id += '_norm'
    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size, total_size=params.total_size, normalize=params.normalize)
    params.dataset = dataset
    model.load_state_dict(torch.load(f'{params.save_dir}/saved_models/{expr_id}.pth'))

    boosting = BoostingOne_Sampler(model=model, dataloader=None, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    boosting.sampling(pred_goal=pred_goal, n_samples=n_samples, data_dim=data_dim, normalize=params.normalize, scaler=scaler, params=params, device=device)






