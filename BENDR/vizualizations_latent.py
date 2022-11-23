import pandas as pd
from matplotlib import pyplot
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt

def plotTSNEandUMAP(TransformerOutputAll, TargetValues, epochNr, dim, foldNr) : # config_myglobals.TransformerOutputAll, [*,*], epoch_nr, 512/2048

    loc_cols = ['loc_' + str(i) for i in range(0,dim)]
    dataFrameLatent = pd.DataFrame(TransformerOutputAll, columns=loc_cols).astype("float16")
    dataFrameLatent.info(memory_usage='deep')

    dataFrameLatent.to_pickle("../LatentSpace/Pickles/dummyLatentEpoch{}_{}_{}.pkl".format(foldNr, epochNr, dim))  
    dataFrameLatent['label'] = TargetValues #[*config_myglobals.TargetOutputsTrain, *config_myglobals.TargetOutputsVal]
    dataFrameLatent.to_pickle("../LatentSpace/Pickles/dummyLatentwithLabel{}_{}_{}.pkl".format(foldNr, epochNr, dim))
    dataFrameLatent['label'] = dataFrameLatent['label'].apply(lambda i: str(i))
    dataFrameLatent.to_pickle("../LatentSpace/Pickles/dummyLatentLabelNames{}_{}_{}.pkl".format(foldNr, epochNr, dim))


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(dataFrameLatent[loc_cols].values)
    df_tsne = dataFrameLatent.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]
    plt.figure(figsize=(16,10))

    sns.scatterplot(
        x="x-tsne", y="y-tsne",
        hue="label",
        palette=sns.color_palette("hls", 2),
        data=df_tsne,
        legend="full",
        alpha=0.3) 

    #plt.show()
    plt.savefig("../LatentSpace/plots/TSNE{}_{}_{}.png".format(foldNr, epochNr, dim))

    #UMAP:

    X, y = dataFrameLatent.drop("label", axis=1), dataFrameLatent[["label"]].values.flatten()

    # Scale
    pipe = make_pipeline(PowerTransformer())
    X = pipe.fit_transform(X.copy())

    # Encode the target to numeric
    y_encoded = pd.factorize(y)[0]

    manifold = umap.UMAP(n_neighbors=50).fit(X)

    X_reduced_2 = manifold.transform(X)

    
    # Plot the results
    plt.figure(figsize=(16,10))
    #plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=y_encoded)

    sns.scatterplot(
        x=X_reduced_2[:, 0], y= X_reduced_2[:, 1],
        hue=y_encoded,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.3) 

    
    #plt.show()
    plt.savefig("../LatentSpace/plots/UMAP{}_{}_{}.png".format(foldNr, epochNr, dim))