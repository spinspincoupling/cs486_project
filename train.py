'''
This script handles the training process.
'''
import preprocess
import parameters
import scipy.io

def train():
    print("Start training...")



def main():
    trainingData, testingData, difficultyLevels, overallScores = preprocess.loadTrainTestData()

    train()


if __name__ == '__main__':
    main()
