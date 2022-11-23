import pandas as pd
from mne.io import read_raw_edf
import mne
import torch

dict = pd.read_pickle("C:/Users/Bex/Desktop/MSc. verkefni/Pickle_backup/EEG_dict00400000458s004_2003_02_1200000458_s004_t001.pkl")
testur = dict['rawData']

print(dict.keys())

print(dict['labeled_windows']["window_1233.996s_1238.996s7"][0])


print(testur.info)

rand = torch.randn(1, 20, 1400) #torch.randn(200, 20, 1536)
rand2 = torch.randn(1, 20, 1536)

## ATH LEIÐ TIL AÐ HÖNDLA TENSOR INPUT MEÐ MISUMANDI DIMENTIONS Í TÍMA
prufaConcat = torch.cat((rand, rand2), 0)

print(prufaConcat.size())

#model = torch.load("C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_pre.pth")

# for name in model._modules.keys():
#      print(name)

#model.eval()


#prufa = model(rand)
