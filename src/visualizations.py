import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


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


def plot_histories(corpus_bleu_history, sentence_bleu_history, loss_history): 
    fig, ax = plt.subplots(1, 3, figsize=(18,4))
    
    ax[0].plot(corpus_bleu_history, sns.xkcd_rgb["denim blue"])
    ax[0].set_title('Corpus BLEU')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('BLEU')
    
    if sentence_bleu_history.shape[1] > 1:
        sns.tsplot(sentence_bleu_history,
                   ax=ax[1],
                   color=sns.xkcd_rgb['faded green'])

    ax[1].set_title('Sentence BLEU')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('BLEU')
    
    ax[2].plot(loss_history, sns.xkcd_rgb["pale red"])
    ax[2].set_title('Average loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Loss')
    
    #plt.subplots_adjust(left=0.2, wspace=0.25, top=0.8)
    return fig


def __plot_heatmap(df, ax, fmt):
    sns.heatmap(df, ax=ax, annot=True, fmt=fmt, cmap="Blues", cbar=False)
    ax.xaxis.tick_top()
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0)
    ax.tick_params(axis='both', labelsize=16)