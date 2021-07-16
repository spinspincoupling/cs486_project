'''
This script handles the training process.
'''
import preprocess
import numpy as np
from os import path
import parameters
import scipy.io
import time
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import nn

from cs486_project.transformer.VTN import VTN

def transformBatch(batch):
    transformations = transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    res = torch.zeros((103,3,224,224))
    indx = 0
    print(batch.shape)
    for data in batch:
        img = Image.fromarray(data)
        res[indx] = transformations(img)
        indx+=1
    print("Transformed Batch size:", res.shape)
    return res


def train(vtn, trainingData, difficultyLevels, overallScores):
    trainingError = []
    print("trainingData:", trainingData.shape)

    optimizer = torch.optim.Adam(vtn.parameters())
    criterion = nn.SmoothL1Loss()

    for epoch in range(parameters.TOTAL_EPOCHS):
        epochTrainError = []
        idx = 0
        ### Training Model
        for miniBatchStart in range(int(trainingData.shape[0]/parameters.TOTAL_EPOCHS)):
            print("MiniBatchStart:",miniBatchStart,"==========================")
            start = time.time()
            mini_train = trainingData[5*miniBatchStart:5*miniBatchStart+5]
            print("mini_train.shape:",mini_train.shape)
            for batch in mini_train:
                batch = transformBatch(batch)
                print("After transformation in train, we have the batch type:", batch.type)
                start = time.time()
                vtn.zero_grad()
                print("Batch Size:", batch.shape)
                output = vtn(batch, difficultyLevels[idx])
                loss = criterion(output, overallScores[idx])
                loss.backward()
                optimizer.step()
                end = time.time()
                print("finish training batch " + str(idx) + " takes: " + str(end - start))
                idx += 1
            loss/= mini_train.shape[0]
            epochTrainError.append(loss)
        epochTrainErrorAvg = sum(epochTrainError)/len(epochTrainError)
        print("Epoch ",epoch," Training Error:", epochTrainErrorAvg, sep=" ")
        trainingError.append(epochTrainErrorAvg)
    return vtn, trainingError

def main():
    trainingData, testingData, difficultyLevels, overallScores = preprocess.loadTrainTestData()
    print("In main")
    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vtn = VTN().to(device)

    train(vtn, trainingData, difficultyLevels, overallScores)


if __name__ == '__main__':

    main()


