# BENDER RUN FILE

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

# Since we are doing a lot of loading, this is nice to suppress some tedious information
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

import pathlib

os.environ["WANDB_API_KEY"] = "f932234e949c1ce38b2a31302750470f64778b70"
os.environ["WANDB_MODE"] = "online"

#TansformerOutput = torch.from_numpy(np.array([]))
#TargetOutputs = np.array([])

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default=".\BENDR\configs\downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default=".\BENDR\configs\metrics.yml", help="Where the listings for config "
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
    print("HELLO")


    #nota þetta til að lesa inn tcav gögnin?
    experiment = ExperimentConfig(args.ds_config)
    if args.results_filename:
        results = ThinkerwiseResultTracker()

    #torch.cuda.init()
    
    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'): #erum bara með eitt dataset
        
        print("DATASET TYPE:")
        print(type(ds))
        #p = pathlib.PurePath('C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/configs/files/S001/S001R01.edf')
        #testerinn = ds._load_raw(p)
        #print(type(testerinn))
        #print(testerinn)

        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config) # bac fyrir mmidb ?
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))): #hver fold með mismunandi train test og val
            
            #tqdm.tqdm.write(torch.cuda.memory_summary())

                            #   WANDB PARAMETERS      #
                #############################
                ############################
            hyper_config = dict(
                epochs=ds.train_params._d['epochs']*5,
                batch_size = ds.train_params._d['batch_size'],
                learning_rate= ds.lr,
                dataset="MMIDB Physionet",
                architecture= args.model)
            
            wandb.init(config = hyper_config, project= args.model)

            config = wandb.config

            print("HELLOOOO")
            print(ds.train_params._d)
            config_myglobals.TotalNrOfEpochs = ds.train_params._d['epochs']
            print("Total Nr of epochs:")
            print(config_myglobals.TotalNrOfEpochs)
            print("fold:")
            print(fold)
            config_myglobals.WhatFold = fold

            print(type(test))
            print(type(training))
            print(type(validation))

            

            if args.model == utils.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu)
            else:
                # without transformer? b
                model = LinearHeadBENDR.from_dataset(training)

            if not args.random_init:
                pass
                updatedParams = model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                              freeze_encoder=args.freeze_encoder)
            
           

        

            wandb.watch(model, log = "all", log_freq = 8)
            process = StandardClassification(model, metrics=added_metrics)
            process.set_optimizer(torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01))

            

            # Fit everything
            
            process.fit(training_dataset=training, validation_dataset=validation, warmup_frac=0.1,
                        retain_best=retain_best, pin_memory=False, **ds.train_params._d)

            ##save model

            # try : 
            #     torch.save(model.state_dict(), "C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_dict_LeftRight.pth")

            #     torch.save(model, "C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_LeftRight.pth")
            # except : 
            #     print("Didn't work to save")
            
            # model.save("C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/save_BENDR_model_LeftRight.pth")

            # #torch.save(model, "C:/Users/Bex/Desktop/MSc. verkefni/BENDR_Code/EEG_Thesis/BENDR/BENDR_pretrained/BENDR_model_LeftRight.pth")

                        

            if args.results_filename:
                if isinstance(test, Thinker):
                    print("test er Thinker!")
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                results.to_spreadsheet(args.results_filename)

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            objgraph.show_backrefs(model, filename='sample-backref-graph.png')
            del model

            print("MODEL DELETED")

            break
            #torch.cuda.synchronize()
            time.sleep(10)



        if args.results_filename:
            results.performance_summary(ds_name)
            results.to_spreadsheet(args.results_filename)

''' 
    print(config_myglobals.TansformerOutput.size())
    loc_cols = ['loc_' + str(i) for i in range(0,512)]
    dataFrameLatent = pd.DataFrame(config_myglobals.TansformerOutput, columns=loc_cols).astype("float16")
    dataFrameLatent.info(memory_usage='deep')

    dataFrameLatent.to_pickle("./dummyLatent.pkl")  
    dataFrameLatent['label'] = [*config_myglobals.TargetOutputsTrain, *config_myglobals.TargetOutputsVal] #config_myglobals.TargetOutputs
    dataFrameLatent.to_pickle("./dummyLatentwithLabel.pkl") 
    dataFrameLatent['label'] = dataFrameLatent['label'].apply(lambda i: str(i))
    dataFrameLatent.to_pickle("./dummyLatentLabelNames.pkl") 


    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(dataFrameLatent[loc_cols].values)

    df_tsne = dataFrameLatent.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]
    df_tsne.to_pickle("./dummyLatentTsne.pkl")

    plt.figure(figsize=(16,10))
    
    sns.scatterplot(
        x="x-tsne", y="y-tsne",
        hue="label",
        palette=sns.color_palette("hls", 2),
        data=df_tsne,
        legend="full",
        alpha=0.3) 
    
    plt.show()
    plt.savefig('TSNELastEpoch.png')

'''