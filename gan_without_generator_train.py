

from common_imports_gan import *

from gan_without_generator_sampling import GAN_Sampler



class GAN_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            loss_dsc=None, 
            loss_gen=None, 
            expr_id=None
        ):
        super(GAN_Model, self).__init__()

        self.loss_dsc = self.get_loss_discriminator(loss_dsc)
        self.loss_gen = self.get_loss_generator(loss_gen)
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        
    def log(self, x):
        
        """custom log function to prevent log of zero(infinity/NaN) problem."""
        return torch.log(torch.max(x, self.eps))
           
    def get_loss_discriminator(self, loss_name):

        log = self.log


        if loss_name in ['stan', 'heur']:

            def loss(real_score, fake_score, rc=0.5, fc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))]
                fake_part = -log(1.0 - fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part
            
        elif loss_name == 'comb':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] 
                fake_part = -log(1.0 - fake_score) 

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part
        elif loss_name == 'comb_mins_fs':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] + log[D(G(z))]
                fake_part = -log(1.0 - fake_score) - ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        elif loss_name == 'comb_plus_fs': # correct

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] -log[D(G(z))]
                fake_part = -log(1.0 - fake_score) + ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        elif loss_name == 'rvrs':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[1/D(x)]
                real_part = 1/real_score

                # max log[1 - D(G(z))] - log[D(G(z))]
                fake_part = -log(1 - fake_score) + ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        return loss
        
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
    
    def train(self, n_epochs=5, lr_dsc=0.001, lr_gen=0.001, lr_fk_dt=0.001, 
              lc_dsc=[0.5, 0.5, 1., 1.], lc_gen=[1., 1.], dsc_inp='fake_data', device='cuda'):

        self.D = self.model.discriminator.to(device)
        self.G = self.model.generator.to(device)
        self.D.train()
        self.G.train()

        self.eps = torch.tensor(1e-6).to(device)   
        fake_sample_grid, self.grid_x, self.grid_y = get_fake_sample_grid(params.grid_size)
        self.fake_sample_grid = fake_sample_grid.to(device)
        self.fake_sample_grid.requires_grad_(True)    

        self.fake_data = nn.Parameter(torch.randn(params.n_samples, data_dim, device=device))

        optimizer_dsc = torch.optim.Adam(self.D.parameters(), lr=lr_dsc) # betas=(0.5, 0.999)
        optimizer_gen = torch.optim.Adam(self.G.parameters(), lr=lr_gen) # betas=(0.9, 0.999), weight_decay=1e-4)
        optimizer_fk_dt = torch.optim.Adam([self.fake_data], lr=lr_fk_dt)

        self.loss_coef_dsc = lc_dsc
        self.loss_coef_gen = lc_gen

        loss_dsc_hist = []
        loss_gen_hist = []
        loss_fk_dt_hist = []
        real_score_hist = []
        fake_score_hist = []
        fake_data_score_hist = []

        row_df_loss = 0
        for k in range(1):
            for itr, x in enumerate(self.dataloader): 

                if dsc_inp=='x_noisy':
                    eps_noise = torch.randn(x.shape[0], data_dim, device=device)      
                    loss_dsc = self.train_discriminator(x, x + 0.1*eps_noise)
                else:
                    loss_dsc = self.train_discriminator(x, self.fake_data.detach())

                optimizer_dsc.zero_grad()
                loss_dsc.backward()
                optimizer_dsc.step()

        epochs = tqdm(range(n_epochs), unit="epoch", mininterval=0, disable=False)
        for epoch in epochs:
            loss_dsc_hist_epoch, loss_gen_hist_epoch, loss_fk_dt_hist_epoch = [], [], []
            # epochs.set_description(f"Epoch {epoch}")
            for itr, x in enumerate(self.dataloader):
                
                # z = torch.randn(x.shape[0], z_dim, device=device)
                # Training discriminator
                for k in range(1):
                    
                    # if dsc_inp=='x_noisy':
                    #     eps_noise = torch.randn(x.shape[0], data_dim, device=device)      
                    #     loss_dsc = self.train_discriminator(x, x + 0.1*eps_noise)
                    # else:
                    loss_dsc = self.train_discriminator(x, self.fake_data.detach())
                
                    optimizer_dsc.zero_grad()
                    loss_dsc.backward()
                    optimizer_dsc.step()

                for k in range(1):
                    
                    
                    eps_noise = torch.randn(x.shape[0], data_dim, device=device)      
                    loss_dsc = self.train_discriminator(x, x + 0.01*eps_noise)

                    optimizer_dsc.zero_grad()
                    loss_dsc.backward()
                    optimizer_dsc.step()

                loss_dsc_hist_epoch.append(loss_dsc.item())
                # Training generator
                for k in range(0):
                    
                    if itr==len(self.dataloader)-1 and (epoch + 1)% params.save_freq == 0 and params.validation:
                        # Compute discriminator output for fake images
                        score_fake = self.D(self.fake_sample_grid)
                        # Compute the generator loss
                        loss_gen = self.loss_gen(score_fake, *self.loss_coef_gen)
                        # Compute gradients with respect to generator output
                        grad_g_z = -torch.autograd.grad(outputs=loss_gen, inputs=self.fake_sample_grid, create_graph=False)[0].detach().cpu().numpy()
                        
                    
                    loss_gen = self.train_generator(z)

                    optimizer_gen.zero_grad()
                    loss_gen.backward()
                    optimizer_gen.step()
                loss_gen = torch.tensor(1)
                loss_gen_hist_epoch.append(loss_gen.item())
                # Optimize fake_data
                for k in range(1):
                    
                    if itr==len(self.dataloader)-1 and (epoch + 1)% params.save_freq == 0 and params.validation:
                        # Compute discriminator output for fake images
                        score_fake = self.D(self.fake_sample_grid)
                        # Compute the generator loss
                        loss_gen = self.loss_gen(score_fake, *self.loss_coef_gen)
                        # Compute gradients with respect to generator output
                        grad_g_z = -torch.autograd.grad(outputs=loss_gen, inputs=self.fake_sample_grid, create_graph=False)[0].detach().cpu().numpy()
                        
                    
                    loss_fk_dt = self.optimize_fake_data(self.fake_data)

                    optimizer_fk_dt.zero_grad()
                    loss_fk_dt.backward()
                    optimizer_fk_dt.step()

                loss_fk_dt_hist_epoch.append(loss_fk_dt.item())
                
            # save progress
            if True: 
                with torch.no_grad():  
                    z = torch.randn(x.shape[0], z_dim, device=device)
                    real_score = self.D(x).mean().item()
                    fake_score = self.D(self.G(z)).mean().item()
                    fake_data_score = self.D(self.fake_data).mean().item()

                real_score_hist.append(real_score)
                fake_score_hist.append(fake_score)
                fake_data_score_hist.append(fake_score)
                loss_dsc_hist.append(np.mean(loss_dsc_hist_epoch))
                loss_gen_hist.append(np.mean(loss_gen_hist_epoch))
                loss_fk_dt_hist.append(np.mean(loss_fk_dt_hist_epoch))

                epochs.set_postfix_str(f"itr: {epoch*len(self.dataloader) + 1:<6}"+
                                       f"|Loss D {loss_dsc_hist[-1]:<6.3f}|Loss G {loss_gen_hist[-1]:.3f}"+
                                       f"|D(x) {real_score:<6.3f}|D(G(z)) {fake_score:<6.3f}")
            
                if params.save_hdf:
                    df_loss_score_per_epoch.loc[row_df_loss, 'epoch'] = epoch + 1
                    # df_loss_score_per_itr.loc[row_df_loss, 'itr'] = itr
                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_dsc_itr'] = loss_dsc_hist_epoch
                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_gen_itr'] = loss_gen_hist_epoch
                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_fk_dt_itr'] = loss_fk_dt_hist_epoch

                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_dsc_epoch'] = np.mean(loss_dsc_hist_epoch)
                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_gen_epoch'] = np.mean(loss_gen_hist_epoch)
                    df_loss_score_per_epoch.loc[row_df_loss, 'loss_fk_dt_epoch'] = np.mean(loss_fk_dt_hist_epoch)
                    df_loss_score_per_epoch.loc[row_df_loss, 'real_score_epoch'] = real_score
                    df_loss_score_per_epoch.loc[row_df_loss, 'fake_score_epoch'] = fake_score
                    df_loss_score_per_epoch.loc[row_df_loss, 'fake_data_score_epoch'] = fake_data_score
                    row_df_loss += 1

                if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                    # print(f'epoch {epoch+1}, loss: {loss.item()}')
                    if params.save_model:
                        torch.save(self.model.state_dict(), f'{params.save_dir}/saved_models/{self.expr_id}.pth')

                if (epoch + 1)% params.save_freq == 0 and params.validation:
                    self.save_result(epoch, self.fake_data.detach().cpu().numpy(), loss_dsc_hist, loss_gen_hist, loss_fk_dt_hist, 
                                     real_score_hist, fake_score_hist, fake_data_score_hist, grad_g_z, x.shape[1], z_dim)
                        
                    self.model.train()
                
        epochs.close()  
        print(f'\n{Fore.YELLOW}Loss D {np.mean(loss_dsc_hist):<6.3f}|Loss G {np.mean(loss_gen_hist):<6.3f}'+
              f'|D(x) {real_score:<6.3f}|D(G(z)) {fake_score:<6.3f}{Fore.RESET}\n')

    def train_discriminator(self, x, z):

        real_data = x
        real_score = self.D(real_data) # D(x)

        # z = torch.randn(x.shape[0], z_dim, device=device)
        # fake_data = self.G(z).detach()
        fake_data = z
        fake_score = self.D(fake_data)  # D(G(z))

        loss = self.loss_dsc(real_score, fake_score, *self.loss_coef_dsc)
        return  loss

    def train_generator(self, z):

        # z = torch.randn(batch_size, z_dim, device=device)
        fake_data = self.G(z)
        fake_score = self.D(fake_data)  # D(G(z))

        loss = self.loss_gen(fake_score, *self.loss_coef_gen)
        return loss

    def optimize_fake_data(self, fake_data):

        # z = torch.randn(batch_size, z_dim, device=device)
        
        fake_score = self.D(fake_data)  # D(G(z))

        loss = self.loss_gen(fake_score, *self.loss_coef_gen)
        return loss

    def save_result(self, epoch, fake_data, loss_dsc_hist, loss_gen_hist, loss_fk_dt_hist, real_score_hist, fake_score_hist, 
                    fake_data_score_hist, grad_g_z, data_dim, z_dim):
        
        sampler = GAN_Sampler(model=self.model, dataloader=self.dataloader, loss_gen=loss_gen, training=True)
        samples, sample_score, data_score, intermediate_smpl, intermediate_grad_g_z, sample_score_hist, data_score_hist = \
            sampler.sampling(n_samples=params.n_samples, data_dim=data_dim, z_dim=z_dim, lr_fk_dt=lr_fk_dt, 
                             lc_dsc=self.loss_coef_dsc, lc_gen=self.loss_coef_gen, n_timestep_smpl=n_timestep_smpl, 
                             grid_size=params.grid_size, normalize=params.normalize, scaler=scaler, device=device)
        sample_score_mean = round(np.mean(sample_score).item(), 4)
        sample_score_std = round(np.std(sample_score).item(), 4)
        data_score_mean = round(np.mean(data_score).item(), 4)
        data_score_std = round(np.std(data_score).item(), 4)
        
        u, v = grad_g_z[:, 0], grad_g_z[:, 1]
        
        # samples = fake_data

        if params.save_fig:
            plot_samples(fake_data, dataset, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}_train_fk_dt.png')
            plot_samples(samples, dataset, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}.png')
            
            stats, labels, colors = [loss_dsc_hist, loss_gen_hist, loss_fk_dt_hist], ['D', 'G', 'FD'], ['C2', 'C3', 'C7']
            file_name = f'{params.save_dir}/plot_samples_training/{self.expr_id}/loss.png'
            title = f'epoch: {epoch+1:<6}| D(x): {data_score_mean:<6}| D(G(z)): {sample_score_mean:<6}'

            save_plot_fig_stats(stats, labels, colors, title, file_name)
            
            stats, labels, colors = [real_score_hist, fake_score_hist, fake_data_score_hist], ['D(x)', 'D(G(z))', 'D(fake_data)'], ['C9', 'C1', 'C6']
            file_name = f'{params.save_dir}/plot_samples_training/{self.expr_id}/score_train_fk_dt.png'
            title = f'epoch: {epoch+1:<6}'
            
            save_plot_fig_stats(stats, labels, colors, title, file_name, xmax=len(fake_score_hist))
            
            stats, labels, colors = [sample_score_hist, data_score_hist], ['D(fake_data)', 'D(x)'], ['C6', 'C9']
            file_name = f'{params.save_dir}/plot_samples_training/{self.expr_id}/score.png'
            title = f'epoch: {epoch+1:<6}|samples=> D(x): {data_score_mean:<6}| D(G(z)): {sample_score_mean:<6}'
            
            save_plot_fig_stats(stats, labels, colors, title, file_name, xmax=len(sample_score_hist))

            plt.figure(dpi=150)
            magnitude = np.hypot(u, v)
            # plt.scatter(self.grid_x, self.grid_y, color='k', s=1)
            plt.scatter(dataset[:, 0], dataset[:, 1], label='data', c='C9', marker='.', s=20)
            plt.scatter(samples[:, 0], samples[:, 1], label='sample', c='C4', marker='.', s=20)
            plt.quiver(self.grid_x, self.grid_y, u, v, magnitude, scale=None, cmap='plasma', pivot='tail', angles='xy', units='width') 
            plt.title(f'epoch: {epoch+1:<6} | '+r'$\frac{\partial{D(G(z))}}{\partial{G(z)}}$')
            plt.savefig(f'{params.save_dir}/plot_samples_training/{self.expr_id}/grad_gen_out_{epoch+1}.png')
            plt.close()


            # save_animation(intermediate_smpl, f'{params.save_dir}/plot_samples_training/{self.expr_id}/sample_{epoch+1}.mp4')
            save_animation_quiver(intermediate_smpl, intermediate_grad_g_z, dataset, self.grid_x, self.grid_y, epoch, f'{params.save_dir}/plot_samples_training/{self.expr_id}/grad_gen_out_{epoch+1}.mp4')
        
        if params.save_hdf:
            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

            dfs = pd.DataFrame(samples, columns=['x', 'y'])
            dfssc = pd.DataFrame(sample_score, columns=['score'])
            dfdsc = pd.DataFrame(data_score, columns=['score'])
            dfsci = pd.DataFrame({'data_score_mean': [data_score_mean], 'data_score_std': [data_score_std], 
                                  'sample_score_mean': [sample_score_mean], 'sample_score_std': [sample_score_std]})
            dfgr = pd.DataFrame({'u': u, 'v': v})
            dffk = pd.DataFrame(samples, columns=['x', 'y'])
            dffksch = pd.DataFrame({'data_score_hist': data_score_hist, 'sample_score_hist': sample_score_hist})
            intermediate_smpl = select_samples_for_plot(intermediate_smpl, n_timestep_smpl, 20)
            dfis = pd.DataFrame(intermediate_smpl, columns=['x', 'y'])
            intermediate_grad_g_z = select_samples_for_plot(intermediate_grad_g_z, n_timestep_smpl, 20)
            u, v = intermediate_grad_g_z[:, 0], intermediate_grad_g_z[:, 1]
            dfig = pd.DataFrame({'u': u, 'v': v})
            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/samples_epoch_{epoch + 1:06}', value=dfs, format='t')
                hdf_store_samples.append(key=f'df/scores_sample_epoch_{epoch + 1:06}', value=dfssc, format='t') # all samples
                hdf_store_samples.append(key=f'df/scores_data_epoch_{epoch + 1:06}', value=dfdsc, format='t') # all data
                hdf_store_samples.append(key=f'df/score_data_sample_info_{epoch + 1:06}', value=dfsci, format='t')
                hdf_store_samples.append(key=f'df/grad_g_z_{epoch + 1:06}', value=dfgr, format='t')
                # fake_data during training
                hdf_store_samples.append(key=f'df/fake_data_epoch_{epoch + 1:06}', value=dffk, format='t')
                # samples generated iteratively
                hdf_store_samples.append(key=f'df/score_data_sample_hist_{epoch + 1:06}', value=dffksch, format='t')
                hdf_store_samples.append(key=f'df/intermediate_smpl_epoch_{epoch + 1:06}', value=dfis, format='t')
                hdf_store_samples.append(key=f'df/intermediate_grad_g_z_{epoch + 1:06}', value=dfig, format='t')

 
            df_loss_score_per_epoch.to_hdf(file_loss, key='key', index=False)


if __name__=='__main__':


    
    method = 'GAN-wo-G'
    
    params = parse_args(method, 'train')
    params.method = method
    set_seed(params.seed)

    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)
 

    model_name = params.model
    data_dim= params.data_dim
    z_dim= params.z_dim

    loss_dsc = params.loss_dsc
    loss_gen = params.loss_gen

    dataset_name = params.dataset
    batch_size = params.batch_size
    total_size = params.total_size
    normalize = params.normalize
    
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

    experiment_info = [
        ['method:', method],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['z_dim:', z_dim],
        ['loss_dsc:', loss_dsc],
        ['loss_gen:', loss_gen],
        ['dataset_name:', dataset_name],
        ['batch_size:', batch_size],
        ['total_size:', total_size],
        ['normalize:', normalize],
        ['n_epochs:', n_epochs],
        ['lr_dsc:' , lr_dsc],
        ['lr_gen:' , lr_gen],
        ['lr_fk_dt:' , lr_fk_dt],
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
    if normalize:
        expr_id += '_norm'
    save_config_json(save_dir, params, expr_id)


    model = select_model_gan(model_info=model_name, data_dim=data_dim, z_dim=z_dim, device=device)
    dataloader, dataset, scaler = select_dataset(dataset_name=dataset_name, batch_size=batch_size, total_size=total_size, normalize=normalize)
    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')
    
    gan = GAN_Model(model=model, dataloader=dataloader, loss_dsc=loss_dsc, loss_gen=loss_gen, expr_id=expr_id)
    print(f'\n {expr_id}\n')

    create_save_dir_training(params.save_dir, expr_id)
    if params.save_hdf:
        file_loss, file_sample, df_loss_score_per_epoch = \
                create_hdf_file_training(params.save_dir, expr_id, dataset, params.n_samples, params.grid_size, params.n_sel_time)

    gan.train(n_epochs=n_epochs, lr_dsc=lr_dsc, lr_gen=lr_gen, lc_dsc=lc_dsc, lc_gen=lc_gen, device=device)

    save_config_json(save_dir, params, expr_id)

   




