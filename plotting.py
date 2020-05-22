from matplotlib import pyplot as plt


def plot_metric(train_values, validation_values=None, xlabel='x', ylabel='y', title='Metric'):
    plt.plot(range(len(train_values)), train_values)

    legend = ['training']
    if validation_values is not None:
        plt.plot(range(len(validation_values)), validation_values)
        legend.append('validation')

    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)

    plt.legend(legend, loc='upper left')
    plt.show()
