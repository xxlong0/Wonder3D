from copy import deepcopy
from time import time
from typing import Any
import torch
from dataclasses import dataclass

from core.opt import MeshOptimizer


@dataclass
class Snapshot:
    step:int
    time:float
    vertices:torch.Tensor #V,3
    faces:torch.Tensor #F,3
    optimizer:Any=None

def snapshot(opt:MeshOptimizer):
    opt = deepcopy(opt)
    opt._vertices.requires_grad_(False)

    return Snapshot(
        step=opt._step,
        time=time()-opt._start,
        vertices=opt.vertices,
        faces=opt.faces,
        optimizer=opt,
    )