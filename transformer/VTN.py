# This implements the VTN AQA model
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import AttnDecoder
from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self.encoder = Encoder(encoder)
        decoder_stacks = []
        for i in range(num_stacks):
            decoder_stacks.append(AttnDecoder())
        self.decoder = nn.Sequential(*decoder_stacks)
        self.fc = nn.Linear(SEQ_SIZE+1, 1)

    def forward(self, clips, difficulty_levels):
        output = self.encoder(clips)
        output = self.decoder(output)
        output = torch.mean(output, 1)
        output = self.fc(torch.cat(output, difficulty_levels, 1))
        return output
