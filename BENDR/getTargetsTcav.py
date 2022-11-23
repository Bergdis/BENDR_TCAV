import torch
import tqdm
import argparse
import objgraph
import time
import utils
from result_tracking import ThinkerwiseResultTracker
from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import Thinker
from dn3.trainable.processes import StandardClassification
from dn3_ext import BENDRClassification, LinearHeadBENDR
import mne
mne.set_log_level(False)
import numpy as np 
from matplotlib import pyplot
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from torchsummary import summary
import config_myglobals
import wandb
import os
import pickle

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="..\BENDR\configs\downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="..\BENDR\configs\metrics.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    args = parser.parse_args()

    experiment = ExperimentConfig(args.ds_config)

    i_0 = 0
    i_1 = 0 

    NumExamp = 20 

    pathToTcavFolders = '.\BENDR\TCAV_folders\'
    

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'): #erum bara með eitt dataset
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config) # bac fyrir mmidb ?
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))): 

                #print(type(test)) #dn3.data.dataset.Dataset
                # print(test.__len__()) 882
                print(test.__getitem__(0)[0].size())
                print(type(test.get_targets()))
                print(test.get_targets())
                # print(len(test.get_targets())) 882
                print(test.get_targets()[0])
                # virkar!! 
                for j in range(0, test.__len__()) : 
                ## creat left and right folder directories
                    sample = test.__getitem__(j)[0] #edf
                    sample = sample[None, :]
                    target = test.get_targets()[j] #target

                    if(i_0 + i_1 == NumExamp*2) : 
                        break

                    elif(target == 0 and i_0 != NumExamp ) :
                        #save to left folder
                        pickleName = 'test_' + str(i_0) + '.pkl'
                        pathToTcavFoldersRight = pathToTcavFolders + "target_left\\"
                        picklePath = os.path.join(pathToTcavFoldersRight, pickleName)
                        # save pickle
                        with open(picklePath, 'wb') as handle:
                            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        i_0 += 1
                        

                    elif(target == 1 and i_1 != NumExamp ) : 
                        #save to right folder
                        pickleName = 'test_' + str(i_1) + '.pkl'
                        pathToTcavFoldersLeft = pathToTcavFolders + "target_right\\"
                        picklePath = os.path.join(pathToTcavFoldersLeft, pickleName)
                        # save pickle
                        with open(picklePath, 'wb') as handle:
                            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        i_1 += 1
                        pass


                    

                

## Get activations

