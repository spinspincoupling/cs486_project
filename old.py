# data loader
import cv2
import os
import parameters
import numpy as np
import scipy.io

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


def loadSingleVideo(path, fileName, outPath):
    # make sure that the fileName can be found under the "path" variable
    # the function will create a folder with name "fileName" under the "outPath" directory and put all the images
    # under this directory. The images are all the frames of the video
    outDir = outPath + fileName[:-4]+"/"
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        cap = cv2.VideoCapture(os.path.join(path, fileName))
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(outDir+ str(i) + '.jpg', frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()

def processImagesForOneVideo(imageFolder):
    res = []
    for i in range(103):
        imagePath = imageFolder+str(i)+".jpg"
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (224, 224))
        res.append(image)

    train_x = np.asarray(res)
    return train_x

def convertVideosToNPArrays(pathOfVideoFiles, videoFileNames, outPathOfJPG):
    '''
    For example, pathOfVideoFile is in "/Users/bzjesus/Documents/CS486/diving", under
    this folder, there are 370 videos
    videoFileNames are defined the parameters, start from "001.avi" to "370.avi"
    outPathOfJPG is "./pics/"

    the function will call loadSingleVideo function and extract 103 frames(JPG files) for each video
    and store the JPG files for each video under the corresponding folder
    For example, the file directory will be /pics/001/0.jpg
                                                     /1.jpg
                                                     /2.jpg...
                                            /pics/002/0.jpg
                                                     /1.jpg
                                                     /2.jpg...
    :param pathOfVideoFile: the path of the video
    :param videoFileNames:
    :param outPathOfJPG:
    :return:
    '''
    pathsOfJPGs = []
    for folderName in videoFileNames:
        pathsOfJPGs.append(outPathOfJPG+str(folderName[:-4])+"/")
        loadSingleVideo(pathOfVideoFiles, folderName, outPathOfJPG)
    print("Finished splitting 370 videos to images\n")
    numpyImages = []
    count = 1
    for imageFolder in pathsOfJPGs:
        numpyImages.append(processImagesForOneVideo(imageFolder))
        count+=1
        if count%100==0:
            print("Finished loading ",count," images",sep="",end="\n")
    print("Finished loading all the images, now we dump that into numpy array")
    numpyImages = np.asarray(numpyImages)
    print("Finished getting all of the data: ", numpyImages.shape,sep="")
    return numpyImages

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
    if not os.path.isdir("./pics/"):
        os.mkdir("./pics/")
    data = convertVideosToNPArrays(parameters.pathOfVideoFiles, generateVideoNames(), "./pics/")
    trainingData = data[trainingIndexs][0]
    testingData = data[testingIndexs][0]
    return trainingData, testingData, difficultyLevels, overallScores







