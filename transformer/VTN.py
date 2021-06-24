# This implements the VTN AQA model
import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
import decoder
from parameters import d_ff, num_stacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VTN(nn.Module):
    def __init__(self):
        super(VTN, self).__init__()
        self.encoder = encoder.Encoder()
        self.decoder = decoder.AttnDecoder()
        self.fc = nn.nn.Linear(d_ff, 1)

    def forward(self, clips, difficulty_levels):
        output = self.encoder.forward(clips)
        for i in range(num_stacks):
            output = self.decoder.forward(output, output, output)
        output = self.fc(torch.cat(output, difficulty_levels, 1))
        return output


def test():
    model = VTN()
    for name, param in model.named_parameters():
        print(name, param.size())
