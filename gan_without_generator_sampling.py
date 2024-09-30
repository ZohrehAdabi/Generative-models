
from common_imports_gan import *


class GAN_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None,
            loss_gen=None,  
            expr_id=None, 
            training=False
        ):
        super(GAN_Sampler, self).__init__()

   
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
       
        
        self.loss_gen = self.get_loss_generator(loss_gen)


    def sampling(self, n_samples=5, data_dim=2, z_dim=2, lr_fk_dt=0.001, lc_dsc=[0.5, 0.5, 1., 1.], lc_gen=[1., 1.], 
                  n_timestep_smpl=40, grid_size=32, normalize=False, scaler=None, params=None, device='cuda'):

        self.D = self.model.discriminator.to(device)
        self.G = self.model.generator.to(device)
        self.D.eval()
        self.G.eval()

        self.eps = eps.to(device) 

        fake_sample_grid, self.grid_x, self.grid_y = get_fake_sample_grid(grid_size)
        self.fake_sample_grid = fake_sample_grid.to(device)
        self.fake_sample_grid.requires_grad_(True)  

        # lr_fk_dt = 0.1
        self.loss_coef_dsc = lc_dsc
        self.loss_coef_gen = lc_gen

        self.fake_data = nn.Parameter(torch.randn(n_samples, data_dim, device=device))
        optimizer_fk_dt = torch.optim.Adam([self.fake_data], lr=lr_fk_dt)

        intermediate_smpl = np.empty([n_timestep_smpl, n_samples, data_dim])
        intermediate_grad_g_z = np.empty([n_timestep_smpl, grid_size**2, data_dim])
        # intermediate_smpl[0] = self.fake_data.data.cpu().numpy()
        sample_score_hist = []

        # Compute discriminator output for fake images
        score_fake = self.D(self.fake_sample_grid)
        # Compute the generator loss
        loss_gen = self.loss_gen(score_fake, *self.loss_coef_gen)
        # Compute gradients with respect to generator output
        grad_g_z = -torch.autograd.grad(outputs=loss_gen, inputs=self.fake_sample_grid, create_graph=False)[0].detach().cpu().numpy()
          
        # Optimize fake_data
        for t in range(n_timestep_smpl):
            

            intermediate_grad_g_z[t, :, :] = grad_g_z

            loss_fk_dt = self.optimize_fake_data(self.fake_data)
            optimizer_fk_dt.zero_grad()
            loss_fk_dt.backward()
            optimizer_fk_dt.step()

            with torch.no_grad():  
                # sample = torch.randn(n_samples, data_dim).to(device)

                # z = torch.randn(n_samples, z_dim, device=device)
                # sample = self.G(z).detach()   

                sample = self.fake_data.detach()
                sample_score = self.D(sample).cpu().numpy()
                sample_score_hist.append(round(np.mean(sample_score), 4))
                # sample_score = self.D(sample).mean().numpy()
                sample = sample.cpu().numpy()
                if normalize:
                    sample = scaler.inverse_transform(sample)

                intermediate_smpl[t, :, :] = sample
            
        sample = intermediate_smpl[-1]

        data_score = []
        with torch.no_grad():  
            for itr, x in enumerate(self.dataloader):
                d_score = self.D(x).cpu().numpy()
                data_score.append(d_score)
        data_score = np.concatenate(data_score)
        # data_score = np.mean(data_score)
        data_score_hist = [round(np.mean(data_score).item(), 4)] * len(sample_score_hist)
        

        self.save_result(sample, sample_score, data_score, params)

        return sample, sample_score, data_score, intermediate_smpl, intermediate_grad_g_z, sample_score_hist, data_score_hist
    
    def log(self, x):
        
        """custom log function to prevent log of zero(infinity/NaN) problem."""
        return torch.log(torch.max(x, self.eps))
    
    def get_loss_generator(self, loss_name):
        
        log = self.log

        if loss_name == 'heur':

            def loss(fake_score, lc=0.5):
                # min -log[D(G(z))]
                loss = -log(fake_score)
                return  lc * torch.mean(loss)
            
        elif loss_name == 'stan':

            def loss(fake_score, lc=0.5):
                # min log[1 - D(G(z))]
                loss = log(1 - fake_score)
                return lc * torch.mean(loss)
            
        elif loss_name in ['comb', 'rvrs']:
            def loss(fake_score, ffc=1., lc=0.5):
                # min log[1 - D(G(z))] - log[D(G(z))]
                loss = log(1 - fake_score) - ffc * log(fake_score)
                return  lc * torch.mean(loss)
            
            
        return loss
    
    def optimize_fake_data(self, fake_data):

        # z = torch.randn(batch_size, z_dim, device=device)
        
        fake_score = self.D(fake_data)  # D(G(z))

        loss = self.loss_gen(fake_score, *self.loss_coef_gen)
        return loss

    def save_result(self, sample, sample_score, data_score, params):

        if not self.training:
            
            create_save_dir(params.save_dir)
            if params.save_fig:
                f_name = f'sample_{self.expr_id}' if params.test_name is None else f'sample_{self.expr_id}_{params.test_name}'
                plot_samples(sample, f'{params.save_dir}/plot_samples/{f_name}.png')
                
               
            if params.save_hdf:
                
                file_sample = create_hdf_file(params.save_dir, self.expr_id, params.dataset, 
                                              params.n_samples, params.test_name)

                import warnings
                warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
                
                sample_score_mean = round(np.mean(sample_score).item(), 4)
                sample_score_std = round(np.std(sample_score).item(), 4)
                data_score_mean = round(np.mean(data_score).item(), 4)
                data_score_std = round(np.std(data_score).item(), 4)

                dfs = pd.DataFrame(sample, columns=['x', 'y'])
                dfssc = pd.DataFrame(sample_score, columns=['score'])
                dfdsc = pd.DataFrame(data_score, columns=['score'])
                dfsci = pd.DataFrame({'data_score_mean': [data_score_mean], 'data_score_std': [data_score_std], 
                                  'sample_score_mean': [sample_score_mean], 'sample_score_std': [sample_score_std]})
                with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                    hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 
                    hdf_store_samples.append(key=f'df/scores_sample', value=dfssc, format='t')
                    hdf_store_samples.append(key=f'df/scores_data', value=dfdsc, format='t')
                    hdf_store_samples.append(key=f'df/score_data_sample_info', value=dfsci, format='t')

if __name__=='__main__':

    method = 'GAN-wo-G'

    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)

    model_name = params.model
    data_dim= params.data_dim
    z_dim= params.z_dim

    loss_dsc = params.loss_dsc
    loss_gen = params.loss_gen

    dataset_name = params.dataset
    n_samples = params.n_samples

    n_epochs = params.n_epoch
    lr_gen = params.lr_gen
    lr_dsc = params.lr_dsc
    lr_fk_dt = params.lr_fk_dt

    lr_gen = params.lr_gen
    lr_dsc = params.lr_dsc

    lc_dsc = [] if not params.lc_dsc else [float(item) if item.strip() not in ['1', '1.0'] else int(item) for item in params.lc_dsc.replace('  ', ' ').strip().split(' ')] # if type of input arg was string
    lc_gen = [] if not params.lc_dsc else [float(item) if item.strip() not in ['1', '1.0'] else int(item)  for item in params.lc_gen.replace('  ', ' ').strip().split(' ')] 

    lc_dsc_id = '_'.join([str(i) for i in lc_dsc]) 
    lc_gen_id = '_'.join([str(i) for i in lc_gen])

    n_timestep_smpl = params.n_timestep_smpl
    params.test_name = None

    experiment_info = [
        ['method:', method],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['z_dim:', z_dim],
        ['loss_dsc:', loss_dsc],
        ['loss_gen:', loss_gen],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['total_size:', params.total_size],
        ['normalize:', params.normalize],
        ['n_epochs:', n_epochs],
        ['lr_gen:' , lr_gen],
        ['lr_dsc:' , lr_dsc],
        ['lr_fk_dt:' , lr_fk_dt],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'GAN-wo-G_{model_name}_{dataset_name}_z_dim_{z_dim}_lr_dsc_{lr_dsc:.0e}_lr_gen_{lr_gen:.0e}_lr_fk_dt_{lr_fk_dt:.0e}_loss_dsc_{loss_dsc}'
    if lc_dsc:
        expr_id += f'_lc_{lc_dsc_id}'

    expr_id += f'_loss_gen_{loss_gen}'
    if lc_dsc:
        expr_id += f'_lc_{lc_gen_id}'
    if params.normalize:
        expr_id += '_norm'
        
    
    model = select_model_gan(model_info=model_name, data_dim=data_dim, z_dim=z_dim, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size, total_size=params.total_size, normalize=params.normalize)
    params.dataset = dataset
    model.load_state_dict(torch.load(f'{params.save_dir}/saved_models/{expr_id}.pth'))

    gan = GAN_Sampler(model=model, dataloader=dataloader, loss_gen=loss_gen, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    gan.sampling(n_samples=n_samples, data_dim=data_dim, z_dim=z_dim, lr_fk_dt=lr_fk_dt, lc_dsc=lc_dsc, lc_gen=lc_gen, 
                 n_timestep_smpl=n_timestep_smpl, grid_size=params.grid_size,  
                 normalize=params.normalize, scaler=scaler, params=params, device=device)






