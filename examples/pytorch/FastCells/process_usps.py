# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Processing the USPS Data. It is assumed that the data is already
# downloaded.

import subprocess
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
import sys

def processData(workingDir, downloadDir):
    def loadLibSVMFile(file):
        data = load_svmlight_file(file)
        features = data[0]
        labels = data[1]
        retMat = np.zeros([features.shape[0], features.shape[1] + 1])
        retMat[:, 0] = labels
        retMat[:, 1:] = features.todense()
        return retMat

    path = f'{workingDir}/{downloadDir}'
    path = os.path.abspath(path)
    trf = f'{path}/train.txt'
    tsf = f'{path}/test.txt'
    assert os.path.isfile(trf), f'File not found: {trf}'
    assert os.path.isfile(tsf), f'File not found: {tsf}'
    train = loadLibSVMFile(trf)
    test = loadLibSVMFile(tsf)
    np.save(f'{path}/train.npy', train)
    np.save(f'{path}/test.npy', test)

if __name__ == '__main__':
    # Configuration
    workingDir = './'
    downloadDir = 'usps10'
    # End config
    print("Processing data")
    processData(workingDir, downloadDir)
    print("Done")
