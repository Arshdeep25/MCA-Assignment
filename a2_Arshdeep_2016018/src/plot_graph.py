import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_spectogram(features):
    print(features.shape)
    plt.figure()
    plt.imshow(features.T, aspect="auto")
    plt.gca().invert_yaxis()


def plot_mfcc(features):
    print(features.shape)
    plt.figure()
    plt.imshow(features, aspect="auto", origin="lower")


train_features = pickle.load(open('Q1_train_X_Submission.pkl', 'rb'))[9500]
plot_spectogram(train_features)

train_features = pickle.load(open('Q2_train_X_Submission.pkl', 'rb'))[9500]
plot_mfcc(train_features)

plt.show()
