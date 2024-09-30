from colorama import Fore
from configs import *
import time
if parallel:
    from subprocess import  Popen
    import sys
   
else:
    from subprocess import run

num_processes = 4   

def run_parallel(cmds_list):
    l = 0
    num_running_process = 0
    process_list = []
    while l < len(cmds_list): # run len(processes) == limit at a time
        
        if num_running_process < num_processes:
            print(Fore.MAGENTA, f'{cmds_list[l]}\n', Fore.RESET)
            proc = Popen(cmds_list[l], stdout=sys.stdout, stderr=sys.stderr, shell=True)
            process_list.append(proc)
            # proc.wait()
            l += 1
            num_running_process += 1
        running_list = list()
        for j in range(len(process_list)):
            p = process_list[j]

            if p.poll() is not  None:
                
                num_running_process -=1
            else:
                running_list.append(p)
        process_list = running_list    
    for p in process_list:
        p.wait() 

seed = 1

normalize = False
save_model = True
validation = True
save_hdf = True
save_fig = False

n_epoch = 5000
total_size = 4096
batch_size = 1024
save_freq = 500


loss_dsc_list = ['stan', 'comb', 'rvrs']
loss_gen_list = ['heur', 'comb', 'rvrs']
i = 2
loss_dsc = loss_dsc_list[i]
loss_gen = loss_gen_list[i]

lc_dsc_stan = ["'0.5 0.5 0.5'", "'0.5 0.5 1'", "'1 0.5 0.5'", "'1 0.5 1'", "'1 1 1'"]
lc_gen_heur = ["0.5", "1"]
lc_dsc_comb = ["'0.5 0.5 0.5'", "'0.5 0.5 1'", "'1 0.5 0.5'", "'1 0.5 1'", "'1 1 1'"]
lc_gen_comb = ["'0.5 0.5'", "'0.25 1'", "'0.5 1'", "'1 1'"]
lc_dsc_rvrs = ["'0.5 0.5 0.5 0.5'", "'0.5 0.5 0.25 1'", "'0.5 0.5 0.5 1'", "'0.5 0.5 1 1'", "'1 0.5 0.5 1'"]
lc_gen_rvrs = ["'0.5 0.5'", "'0.25 1'", "'0.5 1'", "'1 1'"]

if parallel:
    # for l_dsc, l_gen in zip(loss_dsc, loss_dsc):
    if loss_dsc == 'stan' and loss_gen == 'heur':
        lc_dsc = lc_dsc_stan
        lc_gen = lc_gen_heur
    elif loss_dsc == 'comb' and loss_gen == 'comb':
        lc_dsc = lc_dsc_comb
        lc_gen = lc_gen_comb
    elif loss_dsc == 'rvrs' and loss_gen == 'rvrs':
        lc_dsc = lc_dsc_rvrs
        lc_gen = lc_gen_rvrs
    cmds_list = []
    # for lc_d, lc_g in zip(lc_dsc, lc_gen):
    for lc_d in lc_dsc:
        for  lc_g in lc_gen:
            L = ["python", "gan_truly_rkl_train.py", 

            "--model", "ToyGAN_4_64", "--dataset", "2moons",  
            "--z_dim", "2", "--batch_size", f"{batch_size}", "--total_size", f"{total_size}", "--n_epoch", f"{n_epoch}", 
            "--lr_dsc",  "1e-4", "--lr_gen", "1e-4", "--n_samples", "1000", 
            "--loss_dsc",  f"{loss_dsc}", "--loss_gen", f"{loss_gen}",
            "--lc_dsc", f"{lc_d}", "--lc_gen", f"{lc_g}", 
            "--seed", f"{seed}", "--save_freq", f"{save_freq}"
            ]

            
            if normalize: L.append('--normalize')
            if save_model: L.append('--save_model')
            if validation: L.append('--validation')
            if save_hdf: L.append('--save_hdf')
            if save_fig: L.append('--save_fig')

            cmds_list.append(' '.join(L))
        # print(Fore.CYAN, '\n', ' '.join(L), Fore.RESET)
    tic = time.process_time()
    # procs_list = [Popen(cmd, stdout=sys.stdout, stderr=sys.stderr) for cmd in cmds_list]

    run_parallel(cmds_list)
    toc = time.process_time()
    eTime = toc - tic 
    print(f'\nParallel:\nElapsed time during the whole program in seconds: {eTime}')
    print(f'Elapsed time during the whole program in minutes: {eTime/60}\n')

else:
    # for l_dsc, l_gen in zip(loss_dsc, loss_dsc):
    if loss_dsc == 'stan' and loss_gen == 'heur':
        lc_dsc = lc_dsc_stan
        lc_gen = lc_gen_heur
    elif loss_dsc == 'comb' and loss_gen == 'comb':
        lc_dsc = lc_dsc_comb
        lc_gen = lc_gen_comb
    elif loss_dsc == 'rvrs' and loss_gen == 'rvrs':
        lc_dsc = lc_dsc_rvrs
        lc_gen = lc_gen_rvrs
    cmds_list = []
    # for lc_d, lc_g in zip(lc_dsc, lc_gen):
    for lc_d in lc_dsc:
        for  lc_g in lc_gen:
            L = ["python", "gan_truly_rkl_train.py", 

            "--model", "ToyGAN_4_64", "--dataset", "2moons",  
            "--z_dim", "2", "--batch_size", f"{batch_size}", "--total_size", f"{total_size}", "--n_epoch", f"{n_epoch}", 
            "--lr_dsc",  "1e-4", "--lr_gen", "1e-4", "--n_samples", "1000", 
            "--loss_dsc",  f"{loss_dsc}", "--loss_gen", f"{loss_gen}",
            "--lc_dsc", f"{lc_d}",   "--lc_gen", f"{lc_g}", 
            "--seed",f"{seed}", "--save_freq", f"{save_freq}"
            ]

            
            if normalize: L.append('--normalize')
            if save_model: L.append('--save_model')
            if validation: L.append('--validation')
            if save_hdf: L.append('--save_hdf')
            if save_fig: L.append('--save_fig')

    print(Fore.CYAN, '\n', ' '.join(L), Fore.RESET)
    tic = time.process_time()
    run(L) #shell=True
    toc = time.process_time()
    eTime = toc - tic 
    print(f'\nParallel:\nElapsed time during the whole program in seconds: {eTime}')
    print(f'Elapsed time during the whole program in minutes: {eTime/60}\n')