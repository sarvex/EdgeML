# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import os
import numpy as np


def min_max(A, name):
    print(f"{name} has max: {str(np.max(A))} min: {str(np.min(A))}")
    return np.max([np.abs(np.max(A)), np.abs(np.min(A))])


def quantizeBonsaiModels(modelDir, maxValue=127, scalarScaleFactor=1000):
    ls = os.listdir(modelDir)
    paramNameList = []
    paramWeightList = []
    paramLimitList = []

    for file in ls:
        if (
            file.endswith("npy")
            and not file.startswith("mean")
            and not file.startswith("std")
            and not file.startswith("hyperParam")
        ):
            paramNameList.append(file)
            temp = np.load(f"{modelDir}/{file}")
            paramWeightList.append(temp)
            paramLimitList.append(min_max(temp, file))

    paramLimit = np.max(paramLimitList)

    paramScaleFactor = np.round((2.0 * maxValue + 1.0) / (2.0 * paramLimit))

    quantParamWeights = []
    for param in paramWeightList:
        temp = np.round(paramScaleFactor * param)
        temp[temp[:] > maxValue] = maxValue
        temp[temp[:] < -maxValue] = -1 * (maxValue + 1)

        if maxValue <= 127:
            temp = temp.astype('int8')
        elif maxValue <= 32767:
            temp = temp.astype('int16')
        else:
            temp = temp.astype('int32')

        quantParamWeights.append(temp)

    if os.path.isdir(f'{modelDir}/QuantizedPyTorchBonsaiModel') is False:
        try:
            os.mkdir(f'{modelDir}/QuantizedPyTorchBonsaiModel')
            quantModelDir = f'{modelDir}/QuantizedPyTorchBonsaiModel'
        except OSError:
            print(
                f"Creation of the directory {modelDir} failed/QuantizedPyTorchBonsaiModel"
            )

    np.save(
        f"{quantModelDir}/paramScaleFactor.npy",
        paramScaleFactor.astype('int32'),
    )

    for i in range(len(paramNameList)):
        np.save(f"{quantModelDir}/q{paramNameList[i]}", quantParamWeights[i])

    print("\n\nQuantized Model Dir: " + quantModelDir)


def main():
    args = helpermethods.getQuantArgs()
    quantizeBonsaiModels(args.model_dir, int(args.max_val))


if __name__ == '__main__':
    main()
