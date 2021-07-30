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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        batch = batch.to(device)
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
    outputs = []
    criterion = nn.SmoothL1Loss()
    idx = 0

    for batch in testingData:
        loss = 0
        batch = transformBatch(batch)
        batch = batch.to(device)
        output = vtn(batch, torch.unsqueeze(difficultyLevels[idx], 0))
        loss += criterion(output, torch.unsqueeze(overallScores[idx], 0))
        idx += 1
        outputs.append(output.item())
        testError.append(loss.item())
    testErrorAvg = sum(testError) / len(testError)
    print("Testing Error:", testErrorAvg)
    return outputs, testError


def evaluate(path):
    print("Start testing model " + "...")
    testErrors = []
    outputs = []
    # code to load the frozen model
    net = VTN().to(device)
    net.load_state_dict(torch.load(path))
    start = time.time()
    for i in range(14):
        testingData, testingDifficultyLevels, testingOverallScores = preprocess.loadTestData(5 * i, 5 * i + 5)
        testingDifficultyLevels, testingOverallScores = testingDifficultyLevels.to(
            device), testingOverallScores.to(device)
        output, testError = test(net, testingData, testingDifficultyLevels, testingOverallScores)
        testErrors += testError
        outputs += output
        testErrorAvg = sum(testErrors) / len(testErrors)
        print("Average validation error:", testErrorAvg)
    end = time.time()
    _, scores = preprocess.getAllTestResponse()
    MSE = np.square(np.subtract(scores, outputs)).mean()
    print("mean squared error: ", MSE)
    print("finish running test data takes: " + str(end - start))


def main():
    vtn = VTN().to(device)
    optimizer = torch.optim.Adam(vtn.parameters(), lr=parameters.LEARNING_RATE)
    best = 500

    for epoch in range(parameters.TOTAL_EPOCHS):
        print("Training epoch ", epoch)
        epochTrainError = []
        for i in range(60):
            trainingData, trainingDifficultyLevels, trainingOverallScores = preprocess.loadTrainData(5 * i, 5 * i + 5)
            # print("trainingData:", trainingData.shape)
            # print("trainingDifficultyLevels:", trainingDifficultyLevels.size())
            # print("trainingOverallScores:", trainingOverallScores.size())
            trainingDifficultyLevels, trainingOverallScores = trainingDifficultyLevels.to(
                device), trainingOverallScores.to(device)
            vtn, loss = trainOnData(vtn, optimizer, trainingData, trainingDifficultyLevels, trainingOverallScores)
            epochTrainError.append(loss)
            if loss < best:
                best = loss
                if best < 10:
                    torch.save(vtn.state_dict(), parameters.MODEL_PATH)
        epochTrainErrorAvg = sum(epochTrainError) / len(epochTrainError)
        torch.save(vtn.state_dict(), "./frozen_model" + str(epoch) + ".pth")
        print("Epoch ", epoch, " Training Error:", epochTrainErrorAvg, sep=" ")

    testErrors = []
    for i in range(14):
        # print("In main")
        # print("testingData:", trainingData.shape)
        # print("testingDifficultyLevels:", trainingDifficultyLevels.shape)
        # print("testingOverallScores:", trainingOverallScores.shape)
        testingData, testingDifficultyLevels, testingOverallScores = preprocess.loadTestData(5 * i, 5 * i + 5)
        output, testError = test(vtn, testingData, testingDifficultyLevels, testingOverallScores)
        testErrors += testError
    testErrorAvg = sum(testErrors) / len(testErrors)
    print("Average validation error:", testErrorAvg)


if __name__ == '__main__':
    preprocess.getAllTestResponse()
    main()
