import numpy as np
import scipy.io
import torch
import time

from torchvision import transforms
from transformer.VTN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    trainingData = np.load('../data/trainingData.npy')
    testingData = np.load('../data/testingData.npy')
    # _, _, difficultyLevels, overallScores = preprocess.loadMatFiles()
    difficultyLevels = scipy.io.loadmat("../data/diving_difficulty_level.mat")['difficulty_level']
    overallScores = scipy.io.loadmat("../data/diving_overall_scores.mat")['overall_scores']
    # preprocess = transforms.Compose([
    #     transforms.CenterCrop(240),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    trainingData = np.moveaxis(trainingData, -1, 2)
    print("finish preparing data..")
    print(trainingData.shape)
    mini = trainingData[:2]
    mini_train = torch.Tensor(mini)
    difficultyLevels = torch.Tensor(difficultyLevels)
    overallScores = torch.Tensor(overallScores)
    print(mini.shape)
    print(mini_train.size())
    vtn = VTN().to(device)
    optimizer = torch.optim.Adam(vtn.parameters())
    criterion = nn.SmoothL1Loss()
    idx = 0
    for batch in mini_train:
        start = time.time()
        optimizer.zero_grad()
        loss = 0
        output = vtn(batch, difficultyLevels[idx])
        loss += criterion(output, overallScores[idx])
        loss.backward()
        end = time.time()
        print("finish training batch " + str(idx) + " takes: " + str(end - start))
        idx += 1
