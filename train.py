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


# from transformer.VTN import VTN

def transformBatch(batch):
    transformations = transforms.Compose([
        transforms.CenterCrop(240),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    res = torch.zeros((103, 3, 224, 224))
    indx = 0
    for data in batch:
        img = Image.fromarray(data)
        res[indx] = transformations(img)
        indx += 1
    return res


criterion = nn.SmoothL1Loss()


def trainOnData(vtn, optimizer, trainingData, difficultyLevels, overallScores):
    # trainingError = []
    # print()
    # print("Data:", type(trainingData), trainingData.shape)
    # print("Difficulty Levels:", type(difficultyLevels), difficultyLevels.shape)
    # print("OverallScores:", type(overallScores), overallScores.shape)

    # epochTrainError = []
    idx = 0
    ### Training Model
    # for miniBatchStart in range(int(trainingData.shape[0]/parameters.TOTAL_EPOCHS)):
    #     print("MiniBatchStart:",miniBatchStart,"==========================")
    #
    #     mini_train = trainingData[5*miniBatchStart:5*miniBatchStart+5]
    # print("mini_train.shape:", trainingData.shape)
    trainLoss = 0
    for batch in trainingData:
        start = time.time()
        batch = transformBatch(batch)

        vtn.zero_grad()
        output = vtn(batch, torch.unsqueeze(difficultyLevels[idx], 0))
        loss = criterion(output, torch.unsqueeze(overallScores[idx], 0))
        loss.backward()
        optimizer.step()
        end = time.time()
        print("Finish training batch " + str(idx) + " takes: " + str(end - start))
        idx += 1
        trainLoss += loss.item()
    trainLoss /= trainingData.shape[0]
    # epochTrainErrorAvg = sum(epochTrainError)/len(epochTrainError)
    # print("Epoch ",epoch," Training Error:", epochTrainErrorAvg, sep=" ")
    print("Training Error:", trainLoss, sep=" ")
    # trainingError.append(epochTrainErrorAvg)
    return vtn, trainLoss


def test(vtn, testingData, difficultyLevels, overallScores):
    testError = []
    criterion = nn.SmoothL1Loss()
    idx = 0

    for batch in testingData:
        start = time.time()
        loss = 0
        batch = transformBatch(batch)
        # print("Batch Size:", batch.shape)
        output = vtn(batch, torch.unsqueeze(difficultyLevels[idx], 0))
        loss += criterion(output, torch.unsqueeze(overallScores[idx], 0))
        end = time.time()
        print("finish running " + str(idx) + " takes: " + str(end - start))
        idx += 1
        testError.append(loss.item())
    testErrorAvg = sum(testError) / len(testError)
    print("Testing Error over ", testingData.shape[0], " is:", testErrorAvg)
    return vtn, testError


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vtn = VTN().to(device)
    optimizer = torch.optim.Adam(vtn.parameters(), lr=parameters.LEARNING_RATE)
    best = 500
    bestEpoch = 500

    for epoch in range(parameters.TOTAL_EPOCHS):
        print("Training epoch ", epoch)
        epochTrainError = []
        for i in range(60):
            trainingData, trainingDifficultyLevels, trainingOverallScores = preprocess.loadTrainData(5 * i, 5 * i + 5)
            # print("trainingData:", trainingData.shape)
            # print("trainingDifficultyLevels:", trainingDifficultyLevels.size())
            # print("trainingOverallScores:", trainingOverallScores.size())
            vtn, loss = trainOnData(vtn, optimizer, trainingData, trainingDifficultyLevels, trainingOverallScores)
            epochTrainError.append(loss)
            if loss < best:
                best = loss
                if best < 10:
                    torch.save(vtn.state_dict(), parameters.MODEL_PATH)
        epochTrainErrorAvg = sum(epochTrainError) / len(epochTrainError)
        torch.save(vtn.state_dict(), "./frozen_model" + str(epoch) + ".pth")
        print("Epoch ", epoch, " Training Error:", epochTrainErrorAvg, sep=" ")

    # code to load the frozen model
    # net = VTN()
    # net.load_state_dict(torch.load(parameters.MODEL_PATH))

    print("Start testing....");
    for i in range(14):
        # print("In main")
        # print("testingData:", trainingData.shape)
        # print("testingDifficultyLevels:", trainingDifficultyLevels.shape)
        # print("testingOverallScores:", trainingOverallScores.shape)
        testingData, testingDifficultyLevels, testingOverallScores = preprocess.loadTestData(5 * i, 5 * i + 5)
        vtn, testError = test(vtn, testingData, testingDifficultyLevels, testingOverallScores)


if __name__ == '__main__':
    main()
