from common_imports_diffusion import *

from datasets_toy import get_noise_dataset

from diffusion_boosting_sampling import Boosting_Sampler


params = parse_args('train')
set_seed(params.seed)


class Boosting_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            betas=None, 
            n_timesteps=100, 
            expr_id=None
        ):
        super(Boosting_Model, self).__init__()

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

    def train3(self, n_epochs=5, lr=0.01, inner_epoch=100, noise_input=False, gamma=0.025, device='cuda'):

        self.model.to(device)
        self.model.train()

        self.inner_epoch = inner_epoch
        self.loss_fn = nn.MSELoss() # 1/2*torch.mean(torch.sum((predicted_x - x)**2, axis=1), axis=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.dataloader_noise = get_noise_dataset(params.total_size, params.batch_size)

        loss_hist = []
        row_df_loss = 0
        
        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            timesteps = torch.arange(self.n_timesteps-1, -1, -1).to(device) 
                # timesteps = torch.arange(0, self.n_timesteps).to(device) 
            c = 1 / self.n_timesteps
            for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):
                    # noise = torch.randn_like(x_batch).to(device) 
                    # x = x_batch.to(device) 
                predicted_x = noise
                for t in timesteps:   

                    with torch.no_grad():
                        # grad = - (predicted_x - x)
                        grad = - (noise - x)
                    if noise_input:
                        predicted_grad, loss = self.train_new_model2(noise, grad, t, epoch)
                    else:
                        # predicted_grad, loss = self.train_new_model2(predicted_x, grad, t, epoch)
                        predicted_grad, loss = self.train_new_model2(x*c*noise, grad, t, epoch)
                    # with torch.no_grad():
                    #     predicted_x += gamma * predicted_grad

                # loss = self.loss_fn(predicted_x, x)
                loss_hist_epoch.append(loss)

            if params.save_hdf:
                df_loss_per_itr.loc[row_df_loss, 'epoch'] = epoch + 1
                df_loss_per_itr.loc[row_df_loss, 't'] = t
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

            if ((epoch + 1)% params.save_freq==0 or epoch==0) and params.validation:
                self.save_result(epoch, loss_hist, x.shape[1])
                    
                self.model.train()
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):.3f}{Fore.RESET}\n')

    def add_noise_flow(self, x, t, sigma=0.0001):
        
        norm_t = t / self.n_timesteps
        eps_noise = torch.randn_like(x) 
        x_noisy = (1-(1-sigma) * norm_t) * x + eps_noise * norm_t
        grad = -(-(1-sigma) * x + eps_noise)
        return x_noisy, eps_noise, grad 
    def train2(self, n_epochs=5, lr=0.01, inner_epoch=100, noise_input=False, seperated_models=False, gamma=0.025, device='cuda'):

        self.model.to(device)
        self.model.train()

        self.inner_epoch = inner_epoch
        self.loss_fn = nn.MSELoss() # 1/2*torch.mean(torch.sum((predicted_x - x)**2, axis=1), axis=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.dataloader_noise = get_noise_dataset(params.total_size, params.batch_size)

        loss_hist = []
        row_df_loss = 0
        
        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            timesteps = torch.arange(self.n_timesteps-1, -1, -1).to(device) 
                # timesteps = torch.arange(0, self.n_timesteps).to(device) 
            c = 1 / self.n_timesteps
            for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):
                    # noise = torch.randn_like(x_batch).to(device) 
                    # x = x_batch.to(device) 
                predicted_x = noise
                for t in timesteps:   

                    with torch.no_grad():
                        # grad = - (predicted_x - x)
                        grad = - (noise - x)
                    if noise_input:
                        predicted_grad, loss = self.train_new_model(noise, grad, t, epoch, seperated_models)
                    else:
                        predicted_grad, loss = self.train_new_model(predicted_x, grad, t, epoch, seperated_models)
                        # predicted_grad, loss = self.train_new_model((1-c)*x+c*noise, grad, t, epoch, seperated_models)
                    with torch.no_grad():
                        predicted_x += gamma * predicted_grad

                # loss = self.loss_fn(predicted_x, x)
                loss_hist_epoch.append(loss)

            if params.save_hdf:
                df_loss_per_itr.loc[row_df_loss, 'epoch'] = epoch + 1
                df_loss_per_itr.loc[row_df_loss, 't'] = t
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

            if ((epoch + 1)% params.save_freq==0 or (n_epochs-1)==0) and params.validation:
                self.save_result(epoch, loss_hist, x.shape[1], seperated_models)
                    
                self.model.train()
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):.3f}{Fore.RESET}\n')
    
    def train(self, n_epochs=5, lr=0.01, inner_epoch=100, grad_type='pred_x', learner_inp='pred_x', seperated_models=False, gamma=0.025, device='cuda'):

        self.model.to(device)
        self.model.train()

        self.inner_epoch = inner_epoch
        self.loss_fn = nn.MSELoss() # 1/2*torch.mean(torch.sum((predicted_x - x)**2, axis=1), axis=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.dataloader_noise = get_noise_dataset(params.total_size, params.batch_size)

        loss_hist = []
        row_df_loss = 0
        
        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            timesteps = torch.arange(self.n_timesteps-1, -1, -1).to(device) 
                # timesteps = torch.arange(0, self.n_timesteps).to(device) 
            c = 1 / self.n_timesteps
            s_min, s_max = 0.01, 10
            s_max_min = torch.tensor(s_max/s_min) 
            sigma = s_min*(s_max_min)  # sigma(t) = s_min (s_max/s_min)^t)
            sigma_grad = sigma*torch.sqrt(2*torch.log(s_max_min))
            b_min, b_max = 0.0001, 0.02
            beta = (b_max - b_min)
            beta_grad = (b_max - b_min) # b_min + t * (b_max - b_min)
            
            for t in timesteps: 
                # predicted_x = torch.randn(batch_size).to(device) 
                for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):
                    eps_noise = torch.randn_like(x).to(device) 
                    # x = x_batch.to(device) 
                    if t == (self.n_timesteps-1): predicted_x = noise 

                    with torch.no_grad():
                        if grad_type=='pred_x':
                            grad = - (predicted_x - x)
                        elif grad_type=='noise':
                            grad = - (noise - x)
                        elif grad_type=='eps_noise':
                            grad = - (eps_noise - x)
                        # elif grad_type=='eps_noise_sgm':
                        #     grad = - (sigma_grad*eps_noise)
                        # elif grad_type=='eps_noise_ddpm':
                        #     grad = - (eps_noise)
                    if learner_inp=='noise':
                        predicted_grad, loss = self.train_new_model(noise, grad, t, epoch, seperated_models)
                    elif learner_inp=='pred_x':
                        predicted_grad, loss = self.train_new_model(predicted_x, grad, t, epoch, seperated_models)
                    elif learner_inp=='x_noisy':
                        predicted_grad, loss = self.train_new_model((1-c)*x+c*noise, grad, t, epoch, seperated_models)
                    elif learner_inp=='x_eps_noisy':
                        predicted_grad, loss = self.train_new_model((1-c)*x+c*eps_noise, grad, t, epoch, seperated_models)
                    # elif learner_inp=='x_eps_noisy_sgm':
                    #     predicted_grad, loss = self.train_new_model(x+sigma*eps_noise, grad, t, epoch, seperated_models)
                    # elif learner_inp=='x_eps_noisy_ddpm':
                    #     predicted_grad, loss = self.train_new_model(x+sigma*eps_noise, grad, t, epoch, seperated_models)
                    elif learner_inp=='x':
                        predicted_grad, loss = self.train_new_model(x, grad, t, epoch, seperated_models)
                with torch.no_grad():
                    predicted_x += gamma * predicted_grad

                    loss_x = self.loss_fn(predicted_x, x)
                
                # print(f"epoch: {epoch}, t: {t}| Loss-x {loss_x:.3f}| Loss-grd {loss:.3f}")
                epochs.set_postfix_str(f"epoch: {epoch}, t: {t}| Loss-x {loss_x:.3f}| Loss-g {loss:.3f}")
                loss_hist_epoch.append(loss)
                
            # save progress
            if True: 
                loss_hist.append(np.mean(loss_hist_epoch))
                epochs.set_postfix_str(f"itr: {epoch*len(self.dataloader) + 1:<6}| Loss {loss_hist[-1]:<6.3f}")

                if params.save_hdf:
                    df_loss_per_itr.loc[row_df_loss, 'epoch'] = epoch + 1
                    df_loss_per_itr.loc[row_df_loss, 't'] = t
                    df_loss_per_itr.loc[row_df_loss, 'loss_itr'] = loss_hist_epoch
                    df_loss_per_itr.loc[row_df_loss, 'loss_epoch'] = np.mean(loss_hist_epoch)
                    row_df_loss += 1

                if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                    # print(f'epoch {epoch+1}, loss: {loss.item()}')
                    if params.save_model:
                        torch.save(self.model.state_dict(), f'{params.save_dir}/saved_models/{self.expr_id}.pth')

                if ((epoch + 1)% params.save_freq==0 or (n_epochs-1)==0) and params.validation:
                    self.save_result(epoch, loss_hist, x.shape[1], seperated_models)
                        
                    self.model.train()
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):<6.3f}{Fore.RESET}\n')


    def train_new_model2(self, x_noise, grad, t, epoch):
        
        
        epochs_t = tqdm(range(self.inner_epoch), unit="epoch", mininterval=0, disable=False)
        loss_hist = []
        time = t.repeat_interleave(x_noise.shape[0]).view(-1, 1)
        for e in epochs_t:
            predicted_grad = self.model(x_noise, time)
            loss = self.loss_fn(grad, predicted_grad)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()
            loss_hist.append(loss.item())
            epochs_t.set_postfix_str(f"epoch: {epoch}, t: {t}, Loss {np.mean(loss_hist):.3f}")
        epochs_t.close()
        return predicted_grad, np.mean(loss_hist)
    
    def train_new_model(self, x_noise, grad, t, epoch, seperated_models):
        
        
        epochs_t = tqdm(range(self.inner_epoch), unit="epoch", mininterval=0, disable=True)
        loss_hist = []
        time = t if seperated_models else t.repeat_interleave(x_noise.shape[0]).view(-1, 1)
        for e in epochs_t:
            predicted_grad = self.model(x_noise, time)
            loss = self.loss_fn(grad, predicted_grad)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()
            loss_hist.append(loss.item())
            epochs_t.set_postfix_str(f"epoch: {epoch}, t: {t}| Loss-grad {np.mean(loss_hist):.3f}")
        epochs_t.close()
        with torch.no_grad():
            predicted_grad = self.model(x_noise, time)
        return predicted_grad, np.mean(loss_hist)

    def save_result(self, epoch, loss_hist, data_dim, seperated_models):

        sampler = Boosting_Sampler(model=self.model, dataloader=self.dataloader, dataloader_noise=self.dataloader_noise, betas=self.betas, n_timestep_smpl=n_timestep_smpl, training=True)
        samples_zero, intermediate_smpl = sampler.sampling(n_samples=params.n_samples, learner_inp=learner_inp, seperated_models=seperated_models,
                                                           gamma=gamma, data_dim=data_dim, normalize=params.normalize, scaler=scaler, device=device)
      
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

            intermediate_smpl = select_samples_for_plot(intermediate_smpl, self.n_timesteps, params.n_sel_time)
            dfs = pd.DataFrame(intermediate_smpl, columns=['x', 'y'])
            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/samples_epoch_{epoch + 1:06}', value=dfs, format='t')

            dfs = pd.DataFrame(samples_zero, columns=['x', 'y'])
            with pd.HDFStore(file_sample_zero_all, 'a') as hdf_store_samples_zero:
                hdf_store_samples_zero.append(key=f'df/sample_zero_epoch_{epoch + 1:06}', value=dfs, format='t')

            df_loss_per_itr.to_hdf(file_loss, key='key', index=False)

if __name__=='__main__':

    method = 'Boosting'
    
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

    n_epochs = params.n_epoch
    lr = params.lr 
    inner_epoch = params.innr_epoch
    gamma = params.gamma
    learner_inp = params.learner_inp
    grad_type = params.grad_type
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl

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
        ['n_epochs:', n_epochs],
        ['innr_epoch:', inner_epoch],
        ['lr:' , lr],
        ['gamma:', gamma],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')

    expr_id = f'Boosting_T_{n_timesteps}_{model_name}_{dataset_name}_t_dim_{time_dim}' \
                f'_innr_ep_{inner_epoch}_gamma_{gamma}_grad_type_{grad_type}_learner_inp_{learner_inp}'

    if normalize:
        expr_id += '_norm'

    seperated_models = True if 'Sep' in model_name else False
        
    save_config_json(save_dir, params, expr_id)

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=batch_size, total_size=total_size, normalize=normalize)
    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')
    
    boosting = Boosting_Model(model=model, dataloader=dataloader, betas=betas, n_timesteps=n_timesteps, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    """____________________|______________________"""
    params.n_samples = params.total_size 
    create_save_dir_training(params.save_dir, expr_id)
    if params.save_hdf:
        file_loss, file_sample, file_sample_zero_all, df_loss_per_itr = \
                create_hdf_file_training(params.save_dir, expr_id, dataset, n_timestep_smpl,  params.n_sel_time, params.n_samples)

    # boosting.train(n_epochs=n_epochs, lr=lr, inner_epoch=inner_epoch, gamma=gamma, device=device)
    
    boosting.train(n_epochs=n_epochs, lr=lr, inner_epoch=inner_epoch, grad_type=grad_type, learner_inp=learner_inp, 
                   seperated_models=seperated_models, gamma=gamma, device=device)

    save_config_json(save_dir, params, expr_id)
    print(f'\n {expr_id}\n')
   




