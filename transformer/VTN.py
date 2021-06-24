# This implements the VTN AQA model
import torch
import torch.nn as nn
import encoder
import decoder
from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self.encoder = encoder.Encoder()
        decoder_stacks = []
        for i in range(num_stacks):
            decoder_stacks.append(decoder.AttnDecoder())
        self.decoder = nn.Sequential(*decoder_stacks)
        self.fc = nn.Linear(d_embedding+1, 1)

    def forward(self, clips, difficulty_levels):
        output = self.encoder(clips)
        output = self.decoder(output)
        output = self.fc(torch.cat(output, difficulty_levels, 1))
        return output


def test():
    model = VTN()
    for name, param in model.named_parameters():
        print(name, param.size())
