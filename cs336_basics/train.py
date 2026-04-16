from collections.abc import Callable, Iterable
import yaml
from typing import Optional
from functools import partial
import math
import torch
import os
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
from typing import IO, BinaryIO
import numpy.typing as npt
import numpy as np

from train_util import AdamW, get_cosine_learning_rate_sched, \
clip_gradients, get_batch, cross_entropy, save_checkpoint, load_checkpoint
import fire
from langmodel import TransformerLM


def run_train(
    train_data_path: str,
    checkpt_path: str,
    hyperparams: dict,
):
    token_mmap = np.memmap(train_data_path, dtype=np.uint16)
    model_dims = hyperparams['model_dimension']
    adam_params = hyperparams['train_params']['adam_params']
    lr_sched_params = hyperparams['train_params']['lr_schedule']
    device = torch.device('cuda')

    lm = TransformerLM(**model_dims)
    sched = partial(get_cosine_learning_rate_sched, **lr_sched_params)
    opt = AdamW(lm.parameters(), lr=lr_sched_params['a_min'], **adam_params)

    batch_size = hyperparams['train_params']['batch_size']
    grad_accum_steps = hyperparams['train_params']['grad_accum_steps']
    max_steps = hyperparams['train_params']['max_steps']
    grad_max = hyperparams['train_params']['grad_max']

    get_train_batch = partial(
        get_batch, 
        tokens=token_mmap,
        batch_size=batch_size,
        seq_len=model_dims['context_length'],
        device=device
    )

    grad_clip = partial(clip_gradients, params=lm.parameters(), max_norm=grad_max)

    for t in range(1, max_steps+1):
        opt.zero_grad()

        for i in range(grad_accum_steps):
            x, y = get_train_batch()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = lm(x)
                ce = cross_entropy(logits, y)
            
            ce.backward() # accumulates gradient
        grad_clip()
        lr = sched(t=t)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.step()
        print(t, ce.item())

        if t % 5 == 0:
            save_checkpoint(lm, opt, t, checkpt_path)
    return

def run(
    train_data_path: str,
    run_name: str,
):
    with open('./train_configs/experiments.yaml', 'r') as cf:
        configs = yaml.safe_load(cf)
    
    hyperparams = configs[run_name]

    checkpt_path = f'../data/results/{run_name}_checkpoint.pkl'
    run_train(train_data_path=train_data_path, checkpt_path=checkpt_path, hyperparams=hyperparams)
    return

if __name__ == '__main__':
    import fire
    fire.Fire(run)
