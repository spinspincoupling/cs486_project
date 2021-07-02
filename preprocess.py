# data loader
import cv2
import os
import parameters
import numpy as np
import scipy.io
from numpy import save
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
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
    difficultyLevels = scipy.io.loadmat("./data/diving_difficulty_level.mat")['difficulty_level']
    overallScores = scipy.io.loadmat("./data/diving_overall_scores.mat")['overall_scores']
    return trainingIndexs, testingIndexs, difficultyLevels, overallScores


def loadTrainTestData():
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
    trainingIndexs, testingIndexs, difficultyLevels, overallScores = loadMatFiles()
    videoNames = generateVideoNames()
    data = []
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for videoName in videoNames:
        print("Processing",videoName)
        imgs = []
        cap = cv2.VideoCapture(os.path.join(parameters.pathOfVideoFiles, videoName))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = Image.fromarray(frame)
            frame = preprocess(frame)
            temp = tf.keras.preprocessing.image.img_to_array(frame)
            imgs.append(temp)
        data.append(imgs)
    print("Done with all videos, converting to numpy array")
    data = np.asarray(data)
    data = data.reshape(data.shape[0],3, data.shape[1],data.shape[2],data.shape[3])

    data = preprocess(data)
    print("Loading up training and testing")
    trainingData = data[trainingIndexs][0]
    save("train.npy",trainingData)
    testingData = data[testingIndexs][0]
    save("test.npy",testingData)
    print("Returning from loadTrainTestData")

    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)








