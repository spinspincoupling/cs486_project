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
    print("trainingData:", trainingData.shape)
    print("testingData:", testingData.shape)
    print("difficultyLevels:", difficultyLevels.shape)
    print("overallScores:", overallScores.shape)
    train()


if __name__ == '__main__':
    main()
