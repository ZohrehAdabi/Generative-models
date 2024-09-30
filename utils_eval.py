import json 
import torch
import numpy as np
from scipy.linalg import sqrtm

from calssifier_binary_toy_dataset import  get_binary_classifier_for_evaluation
from calssifier_toy_dataset import  get_classifier_for_evaluation
from diffusion_ddpm_sampling import DDPM_Sampler
from beta_scheduling import select_beta_schedule
from models import select_model_diffusion
from datasets_toy import select_dataset

def get_dataset_name_class_num(method, expr_id):
    with open(f'./saved_result/{method}/configs_train/{expr_id}.json', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset']
    num_class = 1
    if dataset_name[0].isdigit():
        num_class = int(dataset_name[0])
        if dataset_name[1].isdigit():
            num_class = int(dataset_name[0:2])
     
    return dataset_name, num_class, config
        
def get_classifiers(dataset_name, num_class, data_size, device='cuda'):
    
    if num_class==2:
        classifier_list = get_binary_classifier_for_evaluation(dataset_name=dataset_name, data_size=data_size, device=device)
    else: 
        classifier_list = get_classifier_for_evaluation(dataset_name=dataset_name, num_class=num_class, data_size=data_size)
    
    return classifier_list

def get_generated_samples(expr_id, n_samples, config, device='cuda'):


    betas = select_beta_schedule(s=config['beta_schedule'], n_timesteps=config['n_timesteps']).to(device)
    dataloader, dataset, scaler = select_dataset(dataset_name=config['dataset'], batch_size=256, total_size=config['total_size'], normalize=config['normalize'])
    model = select_model_diffusion(model_name=config['model'], data_dim=config['data_dim'], time_dim=config['time_dim'], 
                        hidden_dim=config['hidden_dim'], n_timesteps=config['n_timesteps'], device=device).to(device)
    model.load_state_dict(torch.load(f"{config['save_dir']}/{config['dataset']}/saved_models/{expr_id}.pth"))
    
    sampler = DDPM_Sampler(model=model, betas=betas, n_timesteps=config['n_timesteps'], training=True)
    samples, _ = sampler.sampling(n_samples=n_samples, data_dim=config['data_dim'], normalize=config['normalize'], scaler=scaler, device=device)
    samples = samples.to(device)

    return samples

def get_probabilities(samples, classifier_list, num_class):

    class_probability = []
    for classifier in classifier_list:

        if num_class==2:
            prob = classifier(samples).detach().cpu().numpy()
            class_prob = np.concatenate([1-prob, prob], axis=1)
        else:
            pred = classifier(samples)
            class_prob = classifier.probability(pred)

        class_probability.append(class_prob)

    return class_probability

def get_representations(samples, classifier_list, num_class):

    representations = []
    for classifier in classifier_list:

        if num_class==2:
            reprs = classifier.representation(samples).detach().cpu().numpy()
        else: 
            reprs = classifier.representation(samples).detach().cpu().numpy()   
        representations.append(reprs)

    return representations

def calculate_IS(p_yx_list):

    eps = 1e-8
    IS_list = []
    for p_yx in p_yx_list:
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = (p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))).sum(axis=1) # sum over classes
        # average over generated data
        IS = np.exp(kl_d.mean())
        IS_list.append(IS)

    return np.mean(IS_list)

def calculate_FID(h_d_list, h_g_list):

    FID_list = []
    for h_d, h_g in zip(h_d_list, h_g_list):
        μ_1 = np.mean(h_d, axis=0)
        μ_2 = np.mean(h_g, axis=0)
        Σ_1 = np.cov(h_d.T)
        Σ_2 = np.cov(h_g.T)
        diff = μ_1 - μ_2
        FID = np.sum(diff * diff) + np.trace(Σ_1 + Σ_2 - 2 * sqrtm(Σ_1 @ Σ_2))
        FID_list.append(FID)

    return np.mean(FID_list)

def calculate_Mode_score(p_yx_list, num_class):
    eps = 1e-8

    p_data_y = np.ones(num_class) / num_class
    mode_score_list = []
    for p_yx in p_yx_list:
        kl_d_1 = (p_yx * (np.log(p_yx + eps) - np.log(p_data_y + eps))).sum(axis=1)  # sum over classes

        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d_2 = (p_y * (np.log(p_y + eps) - np.log(p_data_y + eps))).sum(axis=1)   # sum over classes
        # average over generated data
        mode_score = np.exp(kl_d_1.mean() - kl_d_2.mean())
        mode_score_list.append(mode_score)
    return np.mean(mode_score_list)

def calculate_RKL(p_yx_list, num_class):
    eps = 1e-8
    p_data_y = np.ones(num_class) / num_class
    rkl_list = []
    for p_yx in p_yx_list:
        # calculate KL divergence using log probabilities
        kl_d = (p_yx * (np.log(p_yx + eps) - np.log(p_data_y + eps))).sum(axis=1) # sum over classes
        # average over generated data
        rkl = np.exp(kl_d.mean())
        rkl_list.append(rkl)
    return np.mean(rkl_list)

def count_missing_modes(samples, num_class):

    if num_class == 8:
        sqrt2 = np.sqrt(2)
        center_xy = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / sqrt2, 1. / sqrt2),
                   (1. / sqrt2, -1. / sqrt2),
                   (-1. / sqrt2, 1. / sqrt2),
                   (-1. / sqrt2, -1. / sqrt2)]) * 2
    else:    
        center_xy = np.array([-1, -.5, 0, .5, 1]) * 2

    centers = []
    for x in center_xy:
        for y in center_xy:
            point = [x, y]
            centers.append(point)
    centers = np.array(centers, dtype='float32') 

    registered_modes, registered_samples = registered_samples_and_modes(samples.cpu().numpy(), centers)
    reg_samples_ratio = registered_samples.shape[0] / samples.shape[0] 
    reg_modes_ratio = registered_modes.shape[0] / centers.shape[0]

    return reg_modes_ratio, reg_samples_ratio

def registered_samples_and_modes(g_data, modes, stddev=0.05):
   
    K = modes.shape[0]
    N = g_data.shape[0]

    dist_to_nearest_mode = np.zeros(N)
    idx_of_nearest_mode = np.zeros(N)

    batch_size = 4096 * 4
    num_minibatches = (N + batch_size - 1) // batch_size
    for n in range(num_minibatches):
        data_batch = g_data[n*batch_size:(n+1)*batch_size,:]
       
        z_batch = np.repeat(data_batch, K, axis=0)

        modes_rep = np.tile(modes, (data_batch.shape[0],1))
        
        d2 = (modes_rep - z_batch) ** 2
        d2 = np.sum(d2, axis=1)
        d2 = d2.reshape((-1,K))
        minVal = np.min(d2, axis=1)
        minIdx = np.argmin(d2, axis=1)
        
        dist_to_nearest_mode[n*batch_size:(n+1)*batch_size] = minVal
        idx_of_nearest_mode[n*batch_size:(n+1)*batch_size] = minIdx

    registered_samples = dist_to_nearest_mode < 3*stddev
    registered_samples_nearest_mode = idx_of_nearest_mode[registered_samples]
    count_registered_samples_in_modes = np.bincount(registered_samples_nearest_mode.astype(int))
            
    registered_modes = np.where(count_registered_samples_in_modes >= 20)[0]

    return registered_modes, registered_samples_nearest_mode

def evaluate_generative_model(method, expr_id, n_samples=2000, data_size=5000, is_score=True, fid_score=True, mode_score=True, rkl_score=True, device='cuda'):

    dataset_name, num_class, config = get_dataset_name_class_num(method, expr_id)
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    samples = get_generated_samples(expr_id, n_samples, config, device)
    classifier_list = get_classifiers(dataset_name, num_class, data_size, device)

    if num_class==1:
        pass
    else:  #num_class>=2:
        if is_score:
            class_probability = get_probabilities(samples, classifier_list, num_class)
        if fid_score:
            dataloader, dataset, scaler = select_dataset(dataset_name, 256, n_samples)
            sample_repres = get_representations(samples, classifier_list, num_class)
            data_repres = get_representations(torch.tensor(dataset).to(device), classifier_list, num_class)

    IS, FID, Mode_score, RKL = None, None, None, None
    if is_score:
        IS = calculate_IS(class_probability)
    if fid_score:
        FID = calculate_FID(data_repres, sample_repres)
    if mode_score:  
        Mode_score = calculate_Mode_score(class_probability, num_class)
    if rkl_score:
        RKL = calculate_RKL(class_probability, num_class)

    reg_modes_ratio, reg_samples_ratio = None, None
    if 'gaussian' in dataset_name:
        reg_modes_ratio, reg_samples_ratio = count_missing_modes(samples, num_class)

    return IS, FID, Mode_score, RKL, reg_modes_ratio, reg_samples_ratio

    
if __name__=='__main__':

    # expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64'
    # IS, FID, Mode_score, RKL, reg_modes_prc, reg_samples_prc = evaluate_generative_model(expr_id, n_samples=2000)
    # print(f"{'Inception score ↑':<16} {IS:.3f}\n{'Frechet score ↓':<16} {FID:.3f}\n{'Mode score ↑':<16} {Mode_score:.3f}\n{'RKL ↓':<16} {RKL:.3f}")

    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    IS, FID, Mode_score, RKL, reg_modes_ratio, reg_samples_ratio = evaluate_generative_model('DDPM', expr_id, n_samples=20000)
   
    print(f"{'reg modes':<16} {reg_modes_ratio:.3f}\n{'reg samples':<16} {reg_samples_ratio}")