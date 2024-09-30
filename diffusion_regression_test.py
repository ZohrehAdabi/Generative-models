from common_imports_diffusion import *

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets_toy import get_noise_dataset
from utils_diffusion import create_save_dir_regression, create_hdf_file_regression, save_animation
from diffusion_regression_test import Regression_test

class Regression_test(nn.Module):

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
        super(Regression_test, self).__init__()

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
    

    def test(self, data_dim=2, normalize=False, scaler=None, device='cuda'):

        self.model.to(device)
        self.model.eval()
        loss_fn = nn.MSELoss()
        error_hist = []
        predictions = []
       
        for itr, (noise, x) in enumerate(zip(self.dataloader_noise, self.dataloader)):

            # noise = torch.randn_like(x_batch).to(device) 
            # x = x_batch.to(device) 
            
            with torch.no_grad():
                predicted_x = self.model(noise)
                loss = loss_fn(predicted_x, x)
                error_hist.append(loss.item())

            predictions.append(predicted_x.cpu().numpy())
        predictions = np.concatenate(predictions)
        error = np.mean(error_hist)
        if not self.training:
            self.save_result(predictions, None, error, params)   
        return predictions, error


    def sampling_boosting_step(self, x, t):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        predicted_grad = self.model(x, t)
        
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

    def save_result(self, predictions, intermediate_smpl, error, params):

        if not self.training:
            create_save_dir_regression(params.save_dir)
            if params.save_fig:
                f_name = f'prediction_{self.expr_id}' if params.test_name is None else f'prediction_{self.expr_id}_{params.test_name}'
                plot_samples(predictions, f'{params.save_dir}/plot_predictions/{f_name}.png')
                save_animation(intermediate_smpl, f'{params.save_dir}/{f_name}.mp4')
            if params.save_hdf:
                n_sel_time = params.n_sel_time if params.n_sel_time <= self.n_timestep_smpl else self.n_timestep_smpl 
                file_prediction = create_hdf_file_regression(params.save_dir, self.expr_id, params.dataset, 
                                              self.n_timestep_smpl, n_sel_time, params.n_samples, params.test_name)
                
                import warnings
                warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                # step = self.n_timestep_smpl // params.n_sel_time
                # dfs = pd.DataFrame(intermediate_smpl[::step, :, :].reshape(-1, data_dim), columns=['x', 'y'])
                if predictions.shape[0] > 1000:
                    predictions = predictions[np.linspace(0, predictions.shape[0]-1, params.n_samples, dtype=int), :]
                
                dfs = pd.DataFrame(predictions, columns=['x', 'y'])          
                dfe = pd.DataFrame([error], columns=['error'])

                with pd.HDFStore(file_prediction, 'a') as hdf_store_prediction:
                    hdf_store_prediction.append(key=f'df/predictions', value=dfs, format='t') 
                    hdf_store_prediction.append(key=f'df/error', value=dfe, format='t')


if __name__=='__main__':

    method = 'Regression'

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
    n_epochs = params.n_epoch
    n_samples = params.n_samples
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
        ['lr:' , params.lr],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'Regression_{model_name}_{dataset_name}'

    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size, total_size=params.total_size, normalize=params.normalize)
    params.dataset = dataset
    model.load_state_dict(torch.load(f'{params.save_dir}/saved_models/{expr_id}.pth'))

    regr = Regression_test(model=model, dataloader=dataloader, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    regr.test(data_dim=data_dim, normalize=params.normalize, scaler=scaler, params=params, device=device)






