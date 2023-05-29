from matplotlib import pyplot as plt


def plot_distribution(df, features):
    fig, axes = plt.subplots(4, 4, figsize=(20, 10))
    axes = axes.flatten()
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, feature in enumerate(features):
        df[feature].plot.hist(ax=axes[i], bins=20)
        axes[i].set_title(feature)
    plt.show()