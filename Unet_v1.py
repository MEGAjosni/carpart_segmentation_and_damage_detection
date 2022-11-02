# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

