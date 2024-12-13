import numpy as np
import torch
import random
import time

def setup_seed(seed=42):
    np.random.seed(42)
    print(f'seed = {seed}')
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def print_current_time():

    timestamp = time.time()

   
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

   
    print("当前时间:", formatted_time)
