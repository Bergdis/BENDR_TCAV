import torch
import numpy as np

TansformerOutput = torch.from_numpy(np.array([]))
TansformerOutputAll = torch.from_numpy(np.array([]))
TargetOutputsTrain = np.array([])       
TargetOutputsVal = np.array([])
WhatEpochIsIt = 0
WhatEpochWasIt = 0
TotalNrOfEpochs = 100000
WhatFold = 0