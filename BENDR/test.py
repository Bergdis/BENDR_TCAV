
import matplotlib.pyplot as plt
from dn3_ext import  LinearHeadBENDR
import torch
from mne.io import read_raw_edf
import mne
from dn3.configuratron.config import RawOnTheFlyRecording, RawTorchRecording, _DumbNamespace
from dn3.transforms.instance import MappingDeep1010, TemporalInterpolation
import numpy as np
from dn3.transforms.instance import To1020
from utils import get_ds
import pandas as pd 

from dn3_ext import LinearHeadBENDR

myModel = LinearHeadBENDR(targets = 2, samples = 1536, channels = 20)

myModel.load("./BENDR/BENDR_pretrained/save_BENDR_model_LeftRight.pth")

rand = torch.randn(200, 20, 1536)

#model = torch.load("C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_LeftRight.pth")

# for name in model._modules.keys():
#      print(name)

#model.eval()


#prufa = model(rand)



#model = dn3_ext.LinearHeadBENDR()

#model.load_state_dict(torch.load("C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_pre.pth"))

#for name, module in model._modules.items():   
#    print(name)
            
#    print(module)

#    print("MODULE NAMES")


'''

Thedict = pd.read_pickle("C:/Users/Bex/Desktop/MSc. verkefni/Pickle_backup/EEG_dict00400000458s004_2003_02_1200000458_s004_t001.pkl")
raw = Thedict['rawData']
print(len(raw))
raw.crop(0,6)
print(len(raw))
#filenames = "C:\\Users\\Bex\\Desktop\\MSc. verkefni\\TUAR\\TUAR\\edf\\01_tcp_ar\\002\\00000254\\s005_2010_11_15\\00000254_s005_t000.edf"
#read_raw_edf(filenames)
#new_raw_one_concept = raw.get_data(tmin = , tmax = )

##PARAMS
sfreq = 250
new_sfreq = 256
tlen = 6
data_max = 3276.7
data_min = -1583.9258304722666

recording = RawTorchRecording(raw, tlen , stride=1, decimate=1, ch_ind_picks=None, bad_spans=None)

#print(recording.size())
#print(recording.__getitem__(1)[0].size())
#print(recording.__getitem__(1)[0])
#print(len(recording.__getitem__(1)))

#print(recording.__len__())

#To1020

print(type(recording.channels))
print(recording.channels)
print(recording.__getitem__(0))

#print(recording.__getitem__(1)[1])

print(recording.__getitem__(0)[0].size())
picks = ['eeg']



_dum = _DumbNamespace(dict(channels=recording.channels, info=dict(data_max=data_max,
                                                                              data_min=data_min)))
xform = MappingDeep1010(_dum, return_mask = True)

#For each item retrieved by __getitem__, transform is called to modify that item.

recording.add_transform(xform)
#recording._execute_transforms()



if sfreq != new_sfreq:
        new_sequence_len = int(tlen * new_sfreq) 
        recording.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=new_sfreq))

print(recording.sfreq)
#print(recording.channels)

# print(recording.__getitem__(1))
# print(len(recording.__getitem__(1)))

# print(recording.__getitem__(1)[0].size())

#For each item retrieved by __getitem__, transform is called to modify that item.

recording.add_transform(To1020())

# print("BÚNA ADDA 1020")

# print(recording.__getitem__(1))
# print(len(recording.__getitem__(1)))

# print(recording.__getitem__(1)[0].size())
# print(recording.__getitem__(1)[0])
# print(recording.__getitem__(1)[1])

# print(len(recording.channels))
# #different_deep1010s = list()

output1020 = recording.__getitem__(0)[0]
output1020 = output1020[None, :]
print(output1020.size())

prufaConcat = torch.cat((output1020, output1020), 0)

prufa = model(output1020)

prufa2 = model(prufaConcat)



def dummy_add_deep1010(self, ch_names: list, deep1010map: np.ndarray, unused):
        for i, (old_names, old_map, unused, count) in enumerate(self._different_deep1010s):
            if np.all(deep1010map == old_map):
                self._different_deep1010s[i] = (old_names, old_map, unused, count+1)
                return
        self._different_deep1010s.append((ch_names, deep1010map, unused, 1))

dummy_add_deep1010([raw.ch_names[i] for i in picks], xform.mapping.numpy(),
                               [raw.ch_names[i] for i in range(len(raw.ch_names)) if i not in picks])

if sfreq != new_sfreq:
        new_sequence_len = int(tlen * new_sfreq) 
        recording.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=new_sfreq))

'''