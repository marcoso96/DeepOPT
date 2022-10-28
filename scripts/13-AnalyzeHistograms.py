import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

acceleration_factor = 22
loss = 'PSNR'
df_pkl_path = 'logs/20-Dictionary_full.pkl'
dataframe = pd.read_pickle(df_pkl_path)
n = 3
print(np.array(dataframe['psnr']['22']['test/psnr_fbp'][0]).mean())
sys.exit(0)
# Subset by part

def plot_boxes(path, plot):

    metrics = ['psnr', 'ssim']
    model_metrics = ['test/{}', 'test/{}_fbp', 'test/{}_admm'] 
    model_metric_labels = ['MoDL - {}', 'FBP - {}', 'ADMM - {}']

    metric_limits = [(5, 45), (0.2, 1.1)]

    for metric, (metric_min, metric_max) in zip(metrics, metric_limits):
        
        fig, axs = plt.subplots(1,1, figsize = (12,8))

        pd_metric = pd.DataFrame()

        for model_metric, model_metric_label in zip(model_metrics, model_metric_labels):

            model_metric = model_metric.format(metric)
            model_metric_label = model_metric_label.format(metric)
             
            dataframe_test = dataframe_x22[[model_metric]]
            pd_metric[model_metric] = dataframe_test[model_metric].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)
        
        if plot == 'violin':
            
            hist = sns.violinplot(data = pd_metric, ax = axs, kde = True)
        
        elif plot == 'box':

            hist = sns.boxplot(data = pd_metric, ax = axs)

        hist.set(xlabel = 'Reconstruction Method', ylabel = metric)
        hist.set_ylim(metric_min, metric_max)
        sns.despine()

        fig.savefig(path.format(metric, acceleration_factor, loss), bbox_inches = 'tight')

def plot_histogram_per_parts(path, plot):

    metrics = ['psnr', 'ssim']
    model_metrics = ['test/{}', 'test/{}_fbp', 'test/{}_admm'] 
    model_metric_labels = ['MoDL - {}', 'FBP - {}', 'ADMM - {}']

    metric_limits = [(5, 45), (0.2, 1.1)]
    features = ['fish_part', 'fish_dpf']
    xlabel_metric = 'PSNR - MoDL'

    for metric, (metric_min, metric_max) in zip(metrics, metric_limits):
        
        fig, axs = plt.subplots(2,len(model_metrics), figsize = (12,8))

        for ax_arr, model_metric, model_metric_label in zip(axs.T, model_metrics, model_metric_labels):

            model_metric = model_metric.format(metric)
            model_metric_label = model_metric_label.format(metric)
            
            for ax, feature in zip(ax_arr, features):
                
                dataframe_test = dataframe[[model_metric, feature]]
        
                pd_metric = pd.DataFrame()
                        
                for part in dataframe_test[feature].unique():

                    series_per_part = dataframe_test[dataframe_test[feature] == part][model_metric].apply(pd.Series).astype(np.float64).stack().reset_index(drop=True)
                    pd_metric[part] = series_per_part
                
                if plot == 'violin':
                    
                    hist = sns.violinplot(data = pd_metric, ax = ax, kde = True)
                
                elif plot == 'box':

                    hist = sns.boxplot(data = pd_metric, ax = ax)

                hist.set(xlabel = model_metric_label, ylabel = metric)
                hist.set_ylim(metric_min, metric_max)
                sns.despine()

        fig.savefig(path.format(metric, acceleration_factor, loss), bbox_inches = 'tight')

if __name__ == '__main__':

    plot = 'box'

    violin_path = 'logs/13-'+plot+'plot_PerParts_normalization_metric-{}-acc_factor_{}-loss_{}.pdf'
    box_path = 'logs/13-'+plot+'plot_normalization_metric-{}-acc_factor_{}-loss_{}.pdf'

    plot_histogram_per_parts(violin_path, 'box')
    plot_boxes(box_path, 'box')