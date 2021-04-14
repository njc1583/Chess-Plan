import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import math
import time
import random

class Ajedrez(nn.Module):
    def __init__(self):
        super(Ajedrez, self).__init__()

        