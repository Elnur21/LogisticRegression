import pandas as pd
import numpy as np
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
        sampleLength = dataset_df.shape[1]
        data_for_each_label = []
        maxClassCount = labels_df.value_counts().max()

        for label in labels_df['Label'].unique():
            classData = dataset_df[labels_df['Label'] == label].values
            classCount = len(classData)
            if(classCount<maxClassCount):
                padding = np.zeros(((maxClassCount-classCount), sampleLength))
                classData = np.concatenate((classData, padding))
            data_for_each_label.append(classData)


        num_rows, num_cols = 4, 5
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
        axes = axes.flatten()
        for j in range(num_rows * num_cols):
            if j < len(dataset_df.columns):
                column_name = dataset_df.columns[j]
                for df in data_for_each_label:
                    data = pd.DataFrame(df).T
                    if data[column_name].any():
                        axes[j].plot(data.index, data[column_name])
                axes[j].set_title(f'Time Series {j+1} of {len(data_for_each_label[0])}')

        plt.tight_layout()
        plt.show()
    except:
        print("Error")


def label_encoder(y):
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in y])
    return encoded_labels


def log(input):
    print(input)