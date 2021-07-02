'''
This script handles the training process.
'''
import preprocess
import numpy as np
from os import path
import parameters
import scipy.io
import torch

def train():
    print("Start training...")


def main():
    trainingData, testingData, difficultyLevels, overallScores = preprocess.loadTrainTestData()
    print("In main")
    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)

    train()


if __name__ == '__main__':

    main()

