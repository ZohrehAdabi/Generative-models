
run_float64 = False
save_dir = './saved_result'

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

parallel = True