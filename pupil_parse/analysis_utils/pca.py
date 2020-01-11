import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sys import platform

import os
import numpy as np
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import glob
import scipy.stats as stats
jtplot.style('grade3', context='poster', fscale=1.4, spines=False, gridlines='--')

def calc_metric_correlations(feature_df, subj_id, reward_code, feature_path=os.path.join(os.path.expanduser('~'),
'Dropbox/loki_0.5/analysis/pupil/processed_data/pupil_features/')):

    feature_corr_matrix_filename = (str(subj_id) + '_' + str(reward_code) + '_feature_corr_matrix.csv')

    features = feature_df.drop(columns=['reward_code', 'subj_id', 'trial'])
    feature_corr_matrix = features.corr(method='spearman')

    feature_corr_matrix.to_csv(os.path.join(feature_path, feature_corr_matrix_filename))

    return feature_corr_matrix, features

def plot_correlogram(feature_df, feature_path=os.path.join(os.path.expanduser('~'),
'Dropbox/loki_0.5/analysis/pupil/processed_data/pupil_features/' )):


    features = feature_df.drop(columns=['reward_code', 'subj_id', 'trial'])
    ax = sns.pairplot(features, kind="scatter")
    ax.fig.suptitle(str(feature_df.subj_id.unique()[0]) + '' + str(feature_df.reward_code.unique()[0]), y=1.08)
    return ax

def pca(feature_df, n_components=None):

    from sklearn.decomposition import PCA

    from sklearn.preprocessing import scale


    X = feature_df.drop(columns=['subj_id', 'reward_code', 'trial'], errors='ignore')

    interp_feature_df = X.interpolate(method='linear')
    assert any(interp_feature_df.isna().sum() == 0), 'check interpolation'

    X_scaled = scale(interp_feature_df)
    assert np.isclose(X_scaled.mean(), 0), 'check centering'
    assert np.isclose(X_scaled.std(), 1), 'check scaling'

    print('mean scaled X ', np.round(X_scaled.mean(), 4), 'std scaled X ', X_scaled.std())

    pca_obj = PCA(n_components=n_components)
    pca_obj.fit(X_scaled) # PCA params stored in object

    pca_projection = pca_obj.fit_transform(X_scaled) # transforms to principal component projections

    print(pca_projection.shape) # should match the interp_feature_df shape
    print(interp_feature_df.shape)

    print(pca_projection[:5])

    return pca_obj, pca_projection, interp_feature_df


def verify_pc_independence(pca_projection):

    print('verifying pc independence')

    print(pca_projection.shape)

    pc_df = pd.DataFrame(pca_projection)
    pc_corr_matrix = pc_df.corr('spearman')

    print('pc_corr_matrix', pc_corr_matrix)


    return pc_corr_matrix

def plot_cumulative_explained_var_ratio(pca_obj, subj_id, reward_code, tag, variance_criterion=0.95,
pupil_metric_fig_path=os.path.join(os.path.expanduser('~'), 'Dropbox/loki_0.5/figures/pupil_pca/')):
    """Plot the cumulative variance explained as principal components are added."""


    fig_name = (str(subj_id) + '_' + str(reward_code) + '_cum_var_ratio_' + str(tag))

    cum_var_explained = np.cumsum(pca_obj.explained_variance_ratio_)
    min_n_components = np.min(np.where(cum_var_explained >= variance_criterion)) + 1

    plt.ioff()
    plt.figure()
    plt.title(fig_name, fontsize=20)
    plt.plot(np.arange(1, len(cum_var_explained)+1), cum_var_explained, '.-')
    plt.plot(min_n_components, cum_var_explained[min_n_components-1], 'ro')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.ylim([0.5, 1])
    plt.savefig(os.path.join(pupil_metric_fig_path, fig_name + '.png'))
    plt.close()

    return min_n_components, cum_var_explained


def plot_eigenvectors(pca_obj, feature_df, subj_id, reward_code,
pupil_metric_fig_path=os.path.join(os.path.expanduser('~'),
'Dropbox/loki_0.5/figures/pupil_pca/')):
    """Plot the magnitude of the eigenvector for each principal component."""

    abs_pca_comps = abs(pca_obj.components_) # [n_components, n_features]
    n_components, n_features = abs_pca_comps.shape[0], abs_pca_comps.shape[1]

    for component in range(n_components):
        fig_name = (str(subj_id) + '_' + str(reward_code) + '_pc' +
        str(component) + '_feature_importance')

        plt.ioff()
        plt.figure()
        plt.title(fig_name, fontsize=20)
        plt.bar(np.arange(0,n_features), abs_pca_comps[component, :]) # for PC_n, feature importance
        plt.xticks(np.arange(0, n_features), labels=feature_df.columns, fontsize=15,
                  rotation=45)
        plt.ylabel('feature importance (eigenvector value)', fontsize=15)
        plt.savefig(os.path.join(pupil_metric_fig_path, fig_name + '.png'))
        plt.close()

    return abs_pca_comps, n_components, n_features


def pc_df(abs_pca_components):
    pc = [abs_pca_components[session][component] for session in range(len(abs_pca_components))]
    pc_df = pd.DataFrame(pc)
    pc_df_melted = pd.melt(pc_df, var_name='feature', value_name='abs_pca_comp')

    return pc_df_melted

def concat_projections_learning_signals(pca_projection_df, learning_signal_df, subj_id,
feature_path=os.path.join(os.path.expanduser('~'),
'Dropbox/loki_0.5/analysis/pupil/processed_data/pupil_features/')):

    projection_ls_filename = (str(subj_id) + '_projection_ls_df.csv')

    learning_signal_df.drop(columns=['index', 'subj_id', 'reward_code', 'trial'], inplace=True)

    projection_ls_df = pd.concat([pca_projection_df, learning_signal_df], axis=1)
    print(projection_ls_df.head())

    projection_ls_df.to_csv(os.path.join(feature_path, projection_ls_filename), index=False)

    return projection_ls_df
