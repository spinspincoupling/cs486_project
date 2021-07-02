'''
This script handles the training process.
'''
import preprocess
import numpy as np
from os import path
import parameters
import scipy.io


def train():
    print("Start training...")


def main():
    if path.exists(parameters.processedTrainingPath) and path.exists(parameters.processedTestingPath):
        _, _, difficultyLevels, overallScores = preprocess.loadMatFiles()
        trainingData = np.load(parameters.processedTrainingPath)
        testingData = np.load(parameters.processedTestingPath)
    else:
        trainingData, testingData, difficultyLevels, overallScores = preprocess.loadTrainTestData()
        # after first run above, save for future uses, DO NOT check in the .npy file, they are too large
        np.save(parameters.processedTrainingPath, trainingData)
        np.save(parameters.processedTestingPath, testingData)
    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)
    train()


if __name__ == '__main__':
    main()
