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

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="..\EEG_Thesis\BENDR\configs\downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="..\EEG_Thesis\BENDR\configs\metrics.yml", help="Where the listings for config "
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

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'): #erum bara með eitt dataset
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config) # bac fyrir mmidb ?
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))): 

            model = LinearHeadBENDR.from_dataset(training)
            
            updatedParams = model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                              freeze_encoder=args.freeze_encoder)

            ##save model
            torch.save(model.state_dict(), "C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_pre.pth")

            torch.save(model, "C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_pre.pth")

            print("MODULE ITEMS")

            for name, module in model._modules.items():
                print(name)
                print(module)

            print("MODULE NAMES")

            for name in model._modules.keys():
                print(name)
                #print(module)

            
            def count_parameters(model):
                table = PrettyTable(["Modules", "Parameters"])
                total_params = 0
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad: continue
                    params = parameter.numel()
                    table.add_row([name, params])
                    total_params+=params
                print(table)
                print(f"Total Trainable Params: {total_params}")
                return total_params
            
            
            #count_parameters(model)

            break

## Get activations



##Get CAV'S

## TEST CAV'S