from common_imports_diffusion import *

from scipy.integrate import solve_ivp

from datasets_toy import get_noise_dataset




class Boosting_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None, 
            dataloader_noise=None,
            betas=None, 
            n_timestep_smpl=100, 
            expr_id=None, 
            training=False
        ):
        super(Boosting_Sampler, self).__init__()

        self.n_timestep_smpl= n_timestep_smpl
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
        
        if dataloader_noise is None:
            self.dataloader_noise = get_noise_dataset(params.total_size, params.batch_size)
        else:
            self.dataloader_noise = dataloader_noise
    
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 
    
    def sampling(self, n_samples=5, gamma=0.025, learner_inp='pred_x', seperated_models=False, data_dim=2, solver='euler', 
                 normalize=False, scaler=None, params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()

        sample = torch.randn(n_samples, data_dim).to(device)
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, data_dim])
        intermediate_smpl[-1] = sample.cpu()
        grads = np.empty([self.n_timestep_smpl, n_samples, data_dim])
        grad_norm_list = []

        if solver=='euler':
            timestp = tqdm(timesteps, disable=self.training)
            
            
            # for itr, noise in enumerate(self.dataloader_noise):
            for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):        
                sample = noise
                # eps_noise = torch.randn_like(x).to(device) 
                noise_count = noise.shape[0]
                for t in timestp: 
                    times = t if seperated_models else torch.repeat_interleave(t, noise_count).reshape(-1, 1).long().to(device)
                    with torch.no_grad():

                        if learner_inp=='noise':
                            predicted_grad = self.sampling_boosting_step_euler(noise, times)
                        elif learner_inp=='pred_x' or learner_inp=='x_noisy' or learner_inp=='x_eps_noisy' or learner_inp=='x_eps_noisy_score':
                            predicted_grad = self.sampling_boosting_step_euler(sample, times)
                        elif learner_inp=='x':
                            predicted_grad = self.sampling_boosting_step_euler(x, times)
                        # elif learner_inp=='x_noisy':
                        #     predicted_grad = self.sampling_boosting_step_euler(sample, times)
                        sample = sample + gamma * predicted_grad
                        grad_norm_list.append([torch.linalg.norm(predicted_grad, axis=1).mean().cpu().numpy()])
                        grads[t, itr*noise_count:(itr+1)*noise_count, :] = predicted_grad.cpu().numpy()
                    if normalize:
                        intermediate_smpl[t, itr*noise_count:(itr+1)*noise_count, :] = scaler.inverse_transform(sample.cpu())
                    else:
                        intermediate_smpl[t, itr*noise_count:(itr+1)*noise_count, :] = sample.cpu()
            timestp.close() 
            sample_zero = intermediate_smpl[0] 

        elif solver=='ode':
        
            with torch.no_grad():
                sample = self.sampling_boosting_step_ode(sample.cpu(), seperated_models).reshape([-1, n_samples, data_dim])
                # sample = sample + gamma * predicted_grad
            if normalize:
                    intermediate_smpl[:-1, :, :] = scaler.inverse_transform(sample)
            else:
                intermediate_smpl[:-1, :, :] = sample
                # intermediate_smpl[t, :, :] = sample.cpu()
            sample = intermediate_smpl[0]
          

        grad_norm = np.concatenate(grad_norm_list)    
        self.save_result(sample_zero, intermediate_smpl, grad_norm, grads, params)

        return sample_zero, intermediate_smpl

    def sampling_(self, n_samples=5, gamma=0.001, data_dim=2, solver='ode', 
                 normalize=False, scaler=None, params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()

        sample = torch.randn(n_samples, data_dim).to(device)
        timesteps = torch.arange(self.n_timesteps-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timesteps+1, n_samples, data_dim])
        intermediate_smpl[-1] = sample.cpu()
        grads = np.empty([self.n_timestep_smpl, n_samples, data_dim])
        grad_norm_list = []

        if solver=='euler':
            timestp = tqdm(timesteps, disable=self.training)
            for t in timestp: 
                times = torch.repeat_interleave(t, n_samples).reshape(-1, 1).long().to(device)
                with torch.no_grad():
                    predicted_grad = self.sampling_boosting_step_euler(sample, times)
                    sample = sample + 0.025 * predicted_grad
                    grad_norm_list.append([torch.linalg.norm(predicted_grad, axis=1).mean().cpu().numpy()])
                    grads[t, :, :] = predicted_grad.cpu().numpy()
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
            sample = intermediate_smpl[0]
          

        grad_norm = np.concatenate(grad_norm_list)    
        self.save_result(sample_zero, intermediate_smpl, grad_norm, grads, params)

        return sample_zero, intermediate_smpl

    def sampling_boosting_step_euler(self, x, t):

        pred_grad = self.model(x, t)
        return pred_grad

    def sampling_boosting_step_ode(self, x, seperated_models):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        solution = solve_ivp(self.ode_func, (1, 0.001), x.reshape([-1,]), args=(self.n_timestep_smpl, seperated_models),
                                                rtol=1e-5, atol=1e-5, method='RK45', dense_output=True)
        # nfe = solution.nfev
        t = np.linspace(0, 1, self.n_timesteps)
        bacward_ode = solution.sol(t).T
        
        return bacward_ode
    
    def ode_func(self, t, x, N, sp):
            
        time = int(t * (N - 1) / 1)
        inp = torch.tensor(x.reshape([-1, 2])).to(torch.float).to(self.model.device)
        times = time if sp else torch.repeat_interleave(time, inp.shape[0]).reshape(-1, 1).long().to(device)
        predicted_grad = - self.model(inp, times).cpu().numpy().reshape([-1,])
        
        return predicted_grad
    
    def compute_mu_theta(self, x, t, predicted_noise):
        """
        DDPM  
          approximated posterior of forward process 
            mu = 1/(1-beta_t) (x_t - beta_t/sqrt(1 - alpha_t) * eps_theta)
        """
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

    def save_result(self, sample, intermediate_smpl, grad_norm, grads, params):

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

                grads = select_samples_for_plot(grads, self.n_timestep_smpl, n_sel_time)
                
                dfn = pd.DataFrame(grads, columns=['x', 'y'])
                dfni = pd.DataFrame({'time': np.arange(len(grad_norm)-1, -1, -1), 'norm': grad_norm})

                with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                    hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 
                    hdf_store_samples.append(key=f'df/grads', value=dfn, format='t')
                    hdf_store_samples.append(key=f'df/grad_info', value=dfni, format='t')

if __name__=='__main__':

    method = 'Boosting'

    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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

    n_epochs = params.n_epoch
    lr = params.lr 
    normalize = params.normalize
    inner_epoch = params.innr_epoch
    gamma = params.gamma
    learner_inp = params.learner_inp
    grad_type = params.grad_type
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl
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
        ['n_epochs:', n_epochs],
        ['innr_epoch:', inner_epoch],
        ['lr:' , lr],
        ['gamma:', gamma],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'Boosting_T_{n_timesteps}_{model_name}_{dataset_name}_t_dim_{time_dim}' \
               f'_innr_ep_{inner_epoch}_gamma_{gamma}_grad_type_{grad_type}_learner_inp_{learner_inp}'
    
    if normalize:
        expr_id += '_norm'

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size, total_size=params.total_size, normalize=params.normalize)
    params.dataset = dataset
    model.load_state_dict(torch.load(f'{params.save_dir}/saved_models/{expr_id}.pth'))

    boosting = Boosting_Sampler(model=model, dataloader=dataloader, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    # boosting.sampling(n_samples=params.n_samples, data_dim=data_dim, normalize=params.normalize, scaler=scaler, params=params, device=device)
    boosting.sampling(n_samples=params.total_size, data_dim=data_dim, learner_inp=learner_inp, 
                       normalize=params.normalize, scaler=scaler, params=params, device=device)






