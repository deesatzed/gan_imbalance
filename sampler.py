import torch
import numpy as np
from copy import deepcopy


def subsample_dataset(train_dataset, label_set):
    """
    :param train_dataset:
    :return: subsampled dataset, where the dataset is imbalance
    """

    # get the feature and labels and transform to numpy
    train_dataset_subsample = deepcopy(train_dataset)
    feature = train_dataset.train_data.numpy()
    labels = train_dataset.train_labels.numpy()

    # get the label set
    labels_set = list(set(labels))
    # chosen_labels = np.random.choice(labels_set, 5, replace=False)
    chosen_labels = [label_set]
    print(chosen_labels)
    feature_resample = []
    labels_resample = []

    for label_idx in labels_set:
        if label_idx in chosen_labels:
            index = np.where(labels == label_idx)[0]
            index = np.random.choice(index, 50, replace=False)

            if len(feature_resample) == 0:
                feature_resample = feature[index]
                labels_resample = labels[index]
            else:
                feature_resample = np.concatenate((feature_resample, feature[index]), axis=0)
                labels_resample = np.concatenate((labels_resample, labels[index]), axis=0)
        # else:
            # index = np.where(labels == label_idx)[0]
            #
            # if len(feature_resample) == 0:
            #     feature_resample = feature[index]
            #     labels_resample = labels[index]
            # else:
            #     feature_resample = np.concatenate((feature_resample, feature[index]), axis=0)
            #     labels_resample = np.concatenate((labels_resample, labels[index]), axis=0)
    feature_resample = torch.from_numpy(feature_resample)
    labels_resample = torch.from_numpy(labels_resample)

    train_dataset_subsample.train_data = feature_resample
    train_dataset_subsample.train_labels = labels_resample
    return train_dataset_subsample


def append_dataset(train_dataset, feature, labels):
    """
    :param train_dataset:
    :return: subsampled dataset, where the dataset is imbalance
    """
    feature_pre = train_dataset.train_data.numpy()
    labels_pre = train_dataset.train_labels.numpy()

    feature_append = np.concatenate((feature_pre, feature), axis=0)
    labels_append = np.concatenate((labels_pre, labels), axis=0)
    feature_append = torch.from_numpy(feature_append)
    labels_append = torch.from_numpy(labels_append)

    train_dataset.train_data = feature_append
    train_dataset.train_labels = labels_append
    return train_dataset

