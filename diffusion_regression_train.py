from common_imports_diffusion import *

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets_toy import get_noise_dataset
from utils_diffusion import save_config_json, create_save_dir_regression_training, create_hdf_file_regression_training
from diffusion_regression_test import Regression_test



class Regression_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            betas=None, 
            n_timesteps=100, 
            expr_id=None
        ):
        super(Regression_Model, self).__init__()

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

    def add_noise(self, x, sigma=0.01):
        
        eps_noise = torch.randn_like(x) 
        x_noisy = x + sigma * eps_noise

        return x_noisy, eps_noise


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


    def train(self, n_epochs=5, lr=0.01, device='cuda'):

        self.model.to(device)
        self.model.train()

        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        loss_hist = []
        row_df_loss = 0
        
        self.dataloader_noise = get_noise_dataset(params.total_size, params.batch_size)

        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        error = 0
        for epoch in epochs:
            loss_hist_epoch = []
            
            for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):
                
                # x = x_batch.to(device) 
                # noise = noise_batch.to(device) 

                predicted_x = self.model(noise)
                loss = loss_fn(x, predicted_x)
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
                loss_hist.append(loss.item())
               
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
                    self.save_result(epoch, loss_hist, x.shape[1])

                    self.model.train()

        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):<6.3f}{Fore.RESET}\n')


    def train_new_model(self, x_noise, grad, t, epoch):
        
        
        epochs_t = tqdm(range(self.inner_epoch), unit="epoch", mininterval=0, disable=False)
        loss_hist = []
        time = t.repeat_interleave(x_noise.shape[0]).view(-1, 1)
        for e in epochs_t:
            predicted_grad = self.model(x_noise, t)
            loss = self.loss_fn(grad, predicted_grad)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()
            loss_hist.append(loss.item())
            epochs_t.set_postfix_str(f"epoch: {epoch}, t: {t}, Loss {np.mean(loss_hist):.3f}")
        epochs_t.close()
        return predicted_grad

    def save_result(self, epoch, loss_hist, data_dim):
        
        regr = Regression_test(model=self.model, dataloader=self.dataloader, dataloader_noise=self.dataloader_noise,
                                       betas=self.betas, n_timestep_smpl=n_timestep_smpl, training=True)
        predictions, error = regr.test(data_dim=data_dim, normalize=params.normalize, scaler=scaler, device=device)
        if params.save_fig:
            plot_samples(predictions, dataset, f'{params.save_dir}/plot_predictions_training/{self.expr_id}/{epoch+1}.png')
            plt.figure()
            plt.plot(loss_hist)
            plt.title(f'epoch: {epoch+1}')
            plt.savefig( f'{params.save_dir}/plot_predictions_training/{self.expr_id}/loss.png')
            plt.close()
        
        if params.save_hdf:
            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            
            if predictions.shape[0] > 1000:
                predictions = predictions[np.linspace(0, predictions.shape[0]-1, params.n_samples, dtype=int), :]
            
            dfs = pd.DataFrame(predictions, columns=['x', 'y'])
            dfe = pd.DataFrame([error], columns=['error'])
            with pd.HDFStore(file_predictions, 'a') as hdf_store_predictions:
                hdf_store_predictions.append(key=f'df/predictions_epoch_{epoch + 1:06}', value=dfs, format='t')
                hdf_store_predictions.append(key=f'df/error_epoch_{epoch + 1:06}', value=dfe, format='t')
                
            df_loss_per_itr.to_hdf(file_loss, key='key', index=False)

if __name__=='__main__':

    method = 'Regression'

    params = parse_args(method, 'train')
    params.method = method
    set_seed(params.seed)
    
    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}'
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

    expr_id = f'Regression_{model_name}_{dataset_name}' 
    if normalize:
        expr_id += '_norm'
               
    save_config_json(save_dir, params, expr_id)

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=batch_size, total_size=total_size, normalize=normalize)
    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')
    
    # X_train, X_test = train_test_split(dataset, train_size=0.8, shuffle=True)
    # dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    # dataloader_test = DataLoader(X_test, batch_size=batch_size, shuffle=True)

    regr = Regression_Model(model=model, dataloader=dataloader, betas=betas, n_timesteps=n_timesteps, expr_id=expr_id)
    print(f'\n {expr_id}\n')

    create_save_dir_regression_training(params.save_dir, expr_id)
    if params.save_hdf:
        file_loss, file_predictions, df_loss_per_itr = \
            create_hdf_file_regression_training(params.save_dir, expr_id, dataset, params.n_timestep_smpl, params.n_sel_time, params.n_samples)

    regr.train(n_epochs=n_epochs, lr=lr, device=device)

    save_config_json(save_dir, params, expr_id)

   




