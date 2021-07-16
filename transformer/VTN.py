# This implements the VTN AQA model
import torch
import torch.nn as nn
from transformer.encoder import Encoder
from transformer.decoder import AttnDecoder
from parameters import *


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self.encoder = Encoder(encoder)
        decoder_stacks = []
        for i in range(num_stacks):
            decoder_stacks.append(AttnDecoder())
        self.decoder = nn.Sequential(*decoder_stacks)
        self.fc = nn.Linear(d_embedding+1, 1)

    def forward(self, clips, difficulty_levels):
        output = self.encoder(clips)
        output = output.unsqueeze(0)
        output = self.decoder(output)
        output = torch.mean(output, 1)
        output = output.squeeze(0)
        print("In VTN:",difficulty_levels.shape)
        output = self.fc(torch.cat((output, difficulty_levels), 0))
        return output
