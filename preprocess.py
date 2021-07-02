# data loader
import cv2
import os
import parameters
import scipy.io
import torchvision.transforms as transforms
from PIL import Image
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
    difficultyLevels = torch.from_numpy(scipy.io.loadmat("./data/diving_difficulty_level.mat")['difficulty_level'])
    overallScores = torch.from_numpy(scipy.io.loadmat("./data/diving_overall_scores.mat")['overall_scores'])

    return trainingIndexs, testingIndexs, difficultyLevels, overallScores

def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255).unsqueeze(0)
def shape(a):
    return (type(a),type(a[0]),type(a[0][0]),len(a),len(a[0]), a[0][0].shape)
def processTrainTestData():
    '''
        data has shape (370, 103, 240, 320, 3)
        370 video data
        103 frames per video
        image 240 * 320 * 3
        data: (370, 103, 240, 320, 3)
        trainingIndexs: (300,)
        testingIndes: (70,)
        difficultyLevels: (370, 1)
        overallScores: (370, 1)
        :return:
        '''
    print(PIL.__version__)
    trainingIndexs, testingIndexs, difficultyLevels, overallScores = loadMatFiles()
    videoNames = generateVideoNames()
    trainingData = []
    testingData = []
    i = 0
    for videoName in videoNames:
        print("Processing",videoName)
        imgs = []
        cap = cv2.VideoCapture(os.path.join(parameters.pathOfVideoFiles, videoName))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = frame.astype("uint8")
            temp = Image.fromarray(frame)
            transformations = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
            processed = transformations(temp)
            imgs.append(processed.numpy())
        if i in trainingIndexs:
            trainingData.append(imgs)
        else:
            testingData.append(imgs)
        i+=1

    print("Done with all videos, converting to numpy array")
    trainingData = torch.from_numpy(np.asarray(trainingData))
    testingData = torch.from_numpy(np.asarray(testingData))

    return trainingData, testingData, difficultyLevels, overallScores


def loadTrainTestData():
    # if path.exists(parameters.processedTrainingPath) and path.exists(parameters.processedTestingPath):
    #     _, _, difficultyLevels, overallScores = loadMatFiles()
    #     trainingData = torch.load(parameters.processedTrainingPath)
    #     testingData = torch.load(parameters.processedTestingPath)
    #     return trainingData, testingData, difficultyLevels, overallScores
    # else:
    trainingData, testingData, difficultyLevels, overallScores = processTrainTestData()
    # after first run above, save for future uses, DO NOT check in the .npy file, they are too large
    print("First Run")
    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)

    # torch.save(trainingData, "data/trainingData.pt")
    # print("Finished saving training data , now we are saving testing data")
    # torch.save(testingData, "data/testingData.pt")

    return trainingData, testingData, difficultyLevels, overallScores





