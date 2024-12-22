from time import perf_counter
import torch

class CPUTimer:
    def __init__(self, prefix='', synchronize=False):
        self.prefix = prefix
        self.synchronize = synchronize

    def __enter__(self):
        if self.synchronize:
            torch.cuda.synchronize()
        self.t = perf_counter()

    def __exit__(self, _type, _value, _traceback):
        if self.synchronize:
            torch.cuda.synchronize()
        t = perf_counter() - self.t
        print('>>>', self.prefix, t, '>>>')
