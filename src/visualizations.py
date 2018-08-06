import constants

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

COLORS = {
    'green': sns.xkcd_rgb["faded green"],
    'red': sns.xkcd_rgb["pale red"],
    'blue': sns.xkcd_rgb["medium blue"],
    'yellow': sns.xkcd_rgb["ochre"]
}


def plot_confusion_dataframe(df, nrows=5, with_percents=False, total=None):
    df = df.head(nrows)
    plt.tight_layout()

    if with_percents:
        assert total != None
        percents = df / total * 100

        fig, ax = plt.subplots(1, 2, figsize=(18,5), sharey=True)
        __plot_heatmap(df, ax=ax[0], fmt="d")
        __plot_heatmap(percents, ax=ax[1], fmt="0.1f")

    else:
        fig, ax = plt.subplots()
        __plot_heatmap(df, ax=ax, fmt="d")

    return fig


def plot_history(history, color, title, ylabel, xlabel='Iteration'): 
    fig, ax = plt.subplots()
    x = constants.LOG_EVERY * np.arange(len(history))
    ax.plot(x, history, color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def __plot_heatmap(df, ax, fmt):
    sns.heatmap(df, ax=ax, annot=True, fmt=fmt, cmap="Blues", cbar=False)
    ax.xaxis.tick_top()
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0)
    ax.tick_params(axis='both', labelsize=16)