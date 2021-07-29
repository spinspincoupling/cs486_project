# data loader
import cv2
import os
import parameters
import scipy.io

import PIL
import torch
import numpy as np
from os import path
def generateVideoNames():
    videoNames = []
    for i in range(1,371):
        videoName = (3-len(str(i))) *"0"+str(i)+".avi"
        videoNames.append(videoName)
    return videoNames
def loadMatFiles():
    '''
    :return:
        trainingIndexs: (1, 300) numpy array
        testingIndexs: (1, 70) numpy array
        difficultyLevels: (370, 1) numpy array
        overallScores: (370, 1) numpy array
    '''
    trainingIndexs = scipy.io.loadmat("./data/split_300_70/training_idx.mat")['training_idx']-1
    testingIndexs = scipy.io.loadmat("./data/split_300_70/testing_idx.mat")['testing_idx']-1

    difficultyLevels = scipy.io.loadmat("./data/diving_difficulty_level.mat")['difficulty_level'].reshape(-1).astype(np.float32)
    overallScores = scipy.io.loadmat("./data/diving_overall_scores.mat")['overall_scores'].reshape(-1).astype(np.float32)
    return trainingIndexs, testingIndexs, difficultyLevels, overallScores

def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255).unsqueeze(0)
def shape(a):
    return (type(a),type(a[0]),type(a[0][0]),len(a),len(a[0]), a[0][0].shape)

def processTrainData(trainIndexs):
    print(PIL.__version__)
    videoNames = generateVideoNames()
    trainingData = []

    for trainIndex in trainIndexs:
        print("Processing",videoNames[trainIndex])
        imgs = []
        path = os.path.join(parameters.pathOfVideoFiles, videoNames[trainIndex])
        cap = cv2.VideoCapture(path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = frame.astype("uint8")
            imgs.append(frame)
        trainingData.append(imgs)
    print("Done with all videos, converting to numpy array")
    trainingData = np.asarray(trainingData)
    return trainingData

def processTestData(testIndexs):
    videoNames = generateVideoNames()
    testingData = []
    for testIndex in testIndexs:
        print("Processing",videoNames[testIndex])
        imgs = []
        path = os.path.join(parameters.pathOfVideoFiles, videoNames[testIndex])
        cap = cv2.VideoCapture(path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = frame.astype("uint8")
            imgs.append(frame)
        testingData.append(imgs)
    print("Done with all videos, converting to numpy array")
    testingData = np.asarray(testingData)
    return testingData

def loadTrainData(trainStart, trainEnd):
    print(PIL.__version__)

    difficultyLevels = scipy.io.loadmat("./data/diving_difficulty_level.mat")['difficulty_level'].reshape(-1).astype(
        np.float32)
    overallScores = scipy.io.loadmat("./data/diving_overall_scores.mat")['overall_scores'].reshape(-1).astype(
        np.float32)
    trainingIndexs = scipy.io.loadmat("./data/split_300_70/training_idx.mat")['training_idx'].reshape(-1) - 1

    trainingIndexs = trainingIndexs[np.arange(trainStart, trainEnd)]

    trainingDifficultyLevels = difficultyLevels[trainingIndexs]
    trainingOverallScores = overallScores[trainingIndexs]

    trainingData = processTrainData(trainingIndexs)
    print("In loadTrainData")
    print("trainingData:", trainingData.shape)
    print("difficultyLevels:", trainingDifficultyLevels.shape)
    print("overallScores:", trainingOverallScores.shape)
    return trainingData, torch.from_numpy(trainingDifficultyLevels),\
        torch.from_numpy(trainingOverallScores)

def loadTestData(testStart, testEnd):
    difficultyLevels = scipy.io.loadmat("./data/diving_difficulty_level.mat")['difficulty_level'].reshape(-1).astype(
        np.float32)
    overallScores = scipy.io.loadmat("./data/diving_overall_scores.mat")['overall_scores'].reshape(-1).astype(
        np.float32)
    testingIndexs = scipy.io.loadmat("./data/split_300_70/testing_idx.mat")['testing_idx'].reshape(-1) - 1

    testingIndexs = testingIndexs[np.arange(testStart, testEnd)]

    testingDifficultyLevels = difficultyLevels[testingIndexs]
    testingOverallScores = overallScores[testingIndexs]

    testingData = processTestData(testingIndexs)
    print("In loadTestData")
    print("testingData:", testingData.shape)
    print("difficultyLevels:", testingDifficultyLevels.shape)
    print("overallScores:", testingOverallScores.shape)
    return testingData, torch.from_numpy(testingDifficultyLevels), \
           torch.from_numpy(testingOverallScores)
