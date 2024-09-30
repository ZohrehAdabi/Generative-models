

from common_imports_gan import *

from models import select_model_gan

from diffusion_ddpm_sampling import DDPM_Sampler





class DDPM_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            betas=None, 
            n_timesteps=100, 
            expr_id=None
        ):
        super(DDPM_Model, self).__init__()

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

    def train(self, n_epochs=5, lr=0.001, device='cuda'):

        self.model.to(device)
        self.model.train()

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)# betas=(0.9, 0.999), weight_decay=1e-4)

        loss_hist = []
        row_df_loss = 0
        
        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            for itr, x_batch in enumerate(self.dataloader):
                
                eps_noise, predicted_noise = self.train_ddpm_step(x_batch)
                loss = loss_fn(eps_noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()

                loss_hist_epoch.append(loss.item())

            if params.save_hdf:
                df_loss_per_itr.loc[row_df_loss, 'epoch'] = epoch + 1
                # df_loss_per_itr.loc[row_df_loss, 'itr'] = itr
                df_loss_per_itr.loc[row_df_loss, 'loss_itr'] = loss_hist_epoch
                    
            if params.save_hdf:
                df_loss_per_itr.loc[row_df_loss, 'loss_epoch'] = np.mean(loss_hist_epoch)
                row_df_loss += 1

            loss_hist.append(np.mean(loss_hist_epoch))
            epochs.set_postfix_str(f"itr: {epoch*len(self.dataloader) + 1}| Loss {loss_hist[-1]:.3f}")

            if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                # print(f'epoch {epoch+1}, loss: {loss.item()}')
                if params.save_model:
                    torch.save(self.model.state_dict(), f'{params.save_dir}/saved_models/{self.expr_id}.pth')

            if (epoch + 1)% params.save_freq == 0 and params.validation:
                self.save_result(epoch, loss_hist, x_batch.shape[1])
                    
                self.model.train()
                
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):.3f}{Fore.RESET}\n')

    def train_ddpm_step(self, x):

        with torch.no_grad():
            t = torch.randint(0, self.n_timesteps, size=[len(x), 1]).to(x.device) # [0, T-1]  beta start from t=1 to T
            # t = torch.randint(0, self.n_timesteps, size=[1]).reshape(x.shape[0], 1).to(x.device)
            x_noisy, eps_noise = self.forward_noising_process(x, t)

        predicted_noise = self.model(x_noisy, t)

        return eps_noise, predicted_noise

    def save_result(self, epoch, loss_hist, data_dim):
        
        sampler = DDPM_Sampler(model=self.model, betas=self.betas, n_timestep_smpl=n_timestep_smpl, training=True)
        samples_zero, intermediate_smpl = sampler.sampling(n_samples=params.n_samples, data_dim=data_dim, 
                                                        normalize=params.normalize, scaler=scaler, device=device)
        if params.save_fig:
            plot_samples(samples_zero, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}.png')
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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    method = 'TrulyRKL'
    
    params = parse_args(method, 'train')
    set_seed(params.seed)

    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)
 

    model_name = params.model
    data_dim= params.data_dim
    z_dim= params.z_dim

    dataset_name = params.dataset
    batch_size = params.batch_size
    total_size = params.total_size
    normalize = params.normalize
    
    n_epochs = params.n_epoch
    lr = params.lr
    lr_disc = params.lr_disc

    experiment_info = [
        ['method:', method],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['z_dim:', z_dim],
        ['dataset_name:', dataset_name],
        ['batch_size:', batch_size],
        ['total_size:', total_size],
        ['normalize:', normalize],
        ['n_epochs:', n_epochs],
        ['lr:' , lr],
        ['lr_disc:' , lr_disc],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')

    expr_id = f'TrulyRKL_beta_{beta_schedule}_T_{n_timesteps}_{model_name}_{dataset_name}_t_dim_{time_dim}'
    if normalize:
        expr_id += '_norm'
    save_config_json(save_dir, params, expr_id)

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_gan(model_info=model_name, data_dim=data_dim, z_dim=z_dim, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=batch_size, total_size=total_size, normalize=normalize)
    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')
    
    ddpm = DDPM_Model(model=model, dataloader=dataloader, betas=betas, n_timesteps=n_timesteps, expr_id=expr_id)
    print(f'\n {expr_id}\n')

    create_save_dir_training(params.save_dir, expr_id)
    if params.save_hdf:
        file_loss, file_sample, file_sample_zero_all, df_loss_per_itr = \
                create_hdf_file_training(params.save_dir, expr_id, dataset, n_timestep_smpl,  params.n_sel_time, params.n_samples)

    ddpm.train(n_epochs=n_epochs, lr=lr, device=device)

    save_config_json(save_dir, params, expr_id)

   




