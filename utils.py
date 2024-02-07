import pandas as pd
from matplotlib import pyplot as plt

from constants import PATH_DATA


def read_dataset(dataset_name):
    datasets_dict = {}
    cur_root_dir = PATH_DATA
    root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'

    df_train = pd.read_csv(root_dir_dataset + dataset_name +
                           '_TRAIN.tsv', sep='\t', header=None)
    df_test = pd.read_csv(root_dir_dataset + dataset_name +
                          '_TEST.tsv', sep='\t', header=None)

    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values
    x_test = x_test.values

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict


def plot(dataset, labels):
    try:
        dataset_df = pd.DataFrame(dataset)
        labels_df = pd.DataFrame(labels, columns=['Label'])
        data_for_each_label = []

        for label in labels_df['Label'].unique():
            data_for_each_label.append(
                dataset_df[labels_df['Label'] == label].values)

        num_rows, num_cols = 3, 3
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
        axes = axes.flatten()
        for j in range(num_rows * num_cols):
            if j < len(dataset_df.columns):
                column_name = dataset_df.columns[j]
                for df in data_for_each_label:
                    data = pd.DataFrame(df).T
                    axes.arrow(2, 5, 3, 3)
                    axes[j].plot(data.index, data[column_name])
                axes[j].set_title(f'Time Series {j+1} of {len(data_for_each_label[0])}')

        plt.tight_layout()
        plt.show()
    except:
        print("salam")
