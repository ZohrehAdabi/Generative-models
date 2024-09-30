from common_imports_diffusion import *

from diffusion_boosting_one_sampling import BoostingOne_Sampler



class BoostingOne_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            betas=None, 
            n_timesteps=100, 
            expr_id=None
        ):
        super(BoostingOne_Model, self).__init__()

        self.n_timesteps= n_timesteps
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        
        
 
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 

    def add_noise_score(self, x, sigma=0.01):
        
        eps_noise = torch.randn_like(x) 
        x_noisy = x + sigma * eps_noise

        return x_noisy, eps_noise

    def score_network(self, score_net, sigma, n_epochs, lr=1e-3, device='cuda'):
        
        score_model = select_model_diffusion(score_net)
        score_model.to(device)
        file = f'{params.save_dir}/saved_models/{self.expr_id}_score_net.pth'
        if os.path.exists():
            score_model.load_state_dict(torch.load(file))
            return score_model.eval()
        else:
            score_model.train()
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(score_model.parameters(), lr=lr)

            loss_hist = []
            epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
            for epoch in epochs:
                loss_hist_epoch = []
                # epochs.set_description(f"Epoch {epoch}")
                for itr, x_batch in enumerate(self.dataloader):
                    x = x_batch.to(device)
                    x_noisy, eps_noise = self.add_noise(x, sigma)

                    predicted_score = score_model(x_noisy)
                    score = - (x_noisy - x) / sigma**2
                    loss = loss_fn(score, predicted_score)

                    # predicted_noise = score_model(x_noisy)
                    # loss = loss_fn(eps_noise, predicted_noise)

                    optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.parameters(), 1)
                    optimizer.step()

                    loss_hist_epoch.append(loss.item())

                loss_hist.append(np.mean(loss_hist_epoch))
                epochs.set_postfix_str(f"itr: {epoch*len(self.dataloader) + 1}| Loss {loss_hist[-1]:.3f}")
            epochs.close()  
            torch.save(score_model.state_dict(), file)
            print(f'\n{Fore.YELLOW}{np.mean(loss_hist):.3f}{Fore.RESET}\n')

        return score_model.eval()

    def forward_noising_process(self, x, t):
        """
        DDPM
        x_{t} = sqrt(alpha_t) * x_{t-1}  +  sqrt(1- alpha_t) * eps
        """
        s_alpha = self.sqrt_alphas[t].reshape(-1, 1)
        s_one_alpha = self.sqrt_one_minus_alphas[t].reshape(-1, 1)
        eps_noise = torch.randn_like(x)
        x_noisy =  s_alpha * x + s_one_alpha * eps_noise

        return x_noisy, eps_noise

    def add_noise_flow_grad(self, x, t, sigma=0.0001):
        
        
        eps_noise = torch.randn_like(x) 
        x_noisy = (1-(1-sigma) * t) * x + eps_noise * t
        # x_noisy = x + eps_noise * t
        grad = -(-(1-sigma) * x + eps_noise)
        return x_noisy, grad 
    
    def add_noise_flow_noise(self, x, t, alpha=0.9996):
        
        eps_noise = torch.randn_like(x) 
        x_noisy = alpha * x + eps_noise * t
    
        return x_noisy, eps_noise
    

    def train(self, pred_goal, n_epochs=5, lr=0.01, inner_epoch=100, gamma=0.001, device='cuda'):

        self.model.to(device)
        self.model.train()

        self.inner_epoch = inner_epoch
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        loss_hist = []
        row_df_loss = 0
        
        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        # noise_dataset = torch.randn([params.total_size, params.data_dim]).to(device) 
        # dataloader_noise = DataLoader(noise_dataset, batch_size=params.batch_size, shuffle=False)
        # timesteps = torch.arange(self.n_timesteps-1, -1, -1).to(device) 
             
        norm_t_noise = 0.025 # beta1
        norm_t_grad = 0.05 # 2/40

        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            
            for itr, x in enumerate(self.dataloader):
                                                    
                # x = x_batch.to(device) 

                if pred_goal=='grad':
                    with torch.no_grad():
                        x_noisy, grad = self.add_noise_flow_grad(x, norm_t_grad)
                    predicted_grad = self.model(x_noisy)
                    loss = self.loss_fn(grad, predicted_grad)
                else: #pred_goal=='noise'
                    with torch.no_grad():
                        x_noisy, eps_noise = self.add_noise_flow_noise(x, norm_t_noise)
                    predicted_noise = self.model(x_noisy)
                    loss = self.loss_fn(eps_noise, predicted_noise)

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss_hist_epoch.append(loss.item())

            # save progress
            if True: 
                loss_hist.append(np.mean(loss_hist_epoch))
                epochs.set_postfix_str(f"itr: {epoch*len(self.dataloader) + 1:<6}| Loss {loss_hist[-1]:<6.3f}")

                if params.save_hdf:
                    df_loss_per_itr.loc[row_df_loss, 'epoch'] = epoch + 1
                    # df_loss_per_itr.loc[row_df_loss, 'itr'] = itr
                    df_loss_per_itr.loc[row_df_loss, 'loss_itr'] = loss_hist_epoch
                    df_loss_per_itr.loc[row_df_loss, 'loss_epoch'] = np.mean(loss_hist_epoch)
                    row_df_loss += 1

                
                if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                    # print(f'epoch {epoch+1}, loss: {loss.item()}')
                    if params.save_model:
                        torch.save(self.model.state_dict(), f'{params.save_dir}/saved_models/{self.expr_id}.pth')

                if (epoch + 1)% params.save_freq == 0 and params.validation:
                    self.save_result(epoch, loss_hist, x.shape[1], pred_goal)
                        
                    self.model.train()
                
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):<6.3f}{Fore.RESET}\n')

    def save_result(self, epoch, loss_hist, data_dim, pred_goal):
        
        sampler = BoostingOne_Sampler(model=self.model, betas=self.betas, n_timestep_smpl=n_timestep_smpl, training=True)
        samples_zero, intermediate_smpl = sampler.sampling(pred_goal=pred_goal, n_samples=params.n_samples, gamma=gamma, data_dim=data_dim,
                                                              normalize=params.normalize, scaler=scaler, device=device)
        if params.save_fig:
            plot_samples(samples_zero, dataset, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}.png')
            plt.figure()
            plt.plot(loss_hist)
            plt.title(f'epoch: {epoch+1}')
            plt.savefig( f'{params.save_dir}/plot_samples_training/{self.expr_id}/loss.png')
            plt.close()
        
        if params.save_hdf:
            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            
            # step = self.n_timesteps // params.n_sel_time
            # dfs = pd.DataFrame(intermediate_smpl[::step, :, :].reshape(-1, data_dim), columns=['x', 'y'])
            intermediate_smpl = select_samples_for_plot(intermediate_smpl, n_timestep_smpl, params.n_sel_time)
            dfs = pd.DataFrame(intermediate_smpl, columns=['x', 'y'])
            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/samples_epoch_{epoch + 1:06}', value=dfs, format='t')
            dfs = pd.DataFrame(samples_zero, columns=['x', 'y'])
            with pd.HDFStore(file_sample_zero_all, 'a') as hdf_store_samples_zero:
                hdf_store_samples_zero.append(key=f'df/sample_zero_epoch_{epoch + 1:06}', value=dfs, format='t')
                
            df_loss_per_itr.to_hdf(file_loss, key='key', index=False)




if __name__=='__main__':

    method = 'BoostingOne' 
     
    params = parse_args(method, 'train')
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

    dataset_name = params.dataset
    batch_size = params.batch_size
    total_size = params.total_size
    normalize = params.normalize

    pred_goal = params.pred_goal
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl

    n_epochs = params.n_epoch
    lr = params.lr 
    inner_epoch = params.innr_epoch
    gamma = params.gamma

    experiment_info = [
        ['method:', method],
        ['beta_schedule:', beta_schedule],
        ['n_timesteps:', n_timesteps],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['time_dim:', time_dim],
        ['dataset_name:', dataset_name],
        ['batch_size:', batch_size],
        ['total_size:', total_size],
        ['normalize:', normalize],
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
                # f'_innr_ep_{inner_epoch}_gamma_{gamma}'
    if normalize:
        expr_id += '_norm'
    save_config_json(save_dir, params, expr_id)

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=batch_size, total_size=total_size, normalize=normalize)
    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')
    
    boosting = BoostingOne_Model(model=model, dataloader=dataloader, betas=betas, n_timesteps=n_timesteps, expr_id=expr_id)
    print(f'\n {expr_id}\n')

    create_save_dir_training(params.save_dir, expr_id)
    if params.save_hdf:
        file_loss, file_sample, file_sample_zero_all, df_loss_per_itr = \
                create_hdf_file_training(params.save_dir, expr_id, dataset, n_timestep_smpl, params.n_sel_time, params.n_samples)

    boosting.train(pred_goal=pred_goal, n_epochs=n_epochs, lr=lr, inner_epoch=inner_epoch, gamma=gamma, device=device)

    save_config_json(save_dir, params, expr_id)
    print(f'{expr_id}')
   




