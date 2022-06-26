import os

from nrdt import get_features_nrdt
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio

emotions = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]
Samples = []
Labels = []
current_label = 0

M = 10
w = 1024
chan = np.array([1, 2, 4, 8, 16, 32, 64])
flag = 0
prag = 0.0001

if __name__ == "__main__":
    emotions = os.listdir("dataset")
    for emotion in emotions:
        for example in os.listdir("dataset/" + emotion):
            features, feat_spec = get_features_nrdt("dataset/" + emotion + "/" + example, M, w, flag, prag, chan)
            if features is not None:
                Samples.append(feat_spec)
                Labels.append(current_label)
        current_label += 1

    vr_set = {
        "Samples": Samples,
        "Labels": Labels

    }

    sio.savemat("emo_dataset.mat", vr_set)

