import numpy as np
import random
import torch
import os

def set_seed(seed: int = 42, verbose: bool = True):
    random.seed(seed)
    py_state = random.getstate()
    py_seed = py_state[1][0]

    np.random.seed(seed)
    np_state = np.random.get_state()
    np_seed = np_state[1][0]


    torch.manual_seed(seed)
    cpu_seed = torch.initial_seed()


    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seed = torch.cuda.initial_seed()
    else:
        cuda_seed = None

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


    info = {
        'PYTHONHASHSEED':        os.environ['PYTHONHASHSEED'],
        'CUBLAS_WORKSPACE_CONFIG': os.environ["CUBLAS_WORKSPACE_CONFIG"],
        'python.random_seed':    py_seed,
        'numpy.random_seed':     np_seed,
        'torch.cpu_seed':        cpu_seed,
        'torch.cuda_seed':       cuda_seed,
    }
    if verbose:
        for k,v in info.items():
            print(f"{k:20s} → {v}")
