import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class AutoencoderWithUncertainty(nn.Module):
    def __init__(self, observation_shape):
        super(AutoencoderWithUncertainty, self).__init__()
        self.observation_shape = observation_shape
        self.feat_extract = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
            ),
            nn.Tanh(),
        )
        self.forward_predictor = nn.Sequential(
            nn.Linear(621, 147), nn.Tanh(), nn.Linear(147, 147)
        )
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(621, 147), nn.Tanh(), nn.Linear(147, 147)
        )

    def forward(self, inputs, action_vector):
        inputs_perm = inputs.permute(0, 3, 1, 2)
        embedding = self.feat_extract(inputs_perm.float())
        embedding = embedding.flatten(start_dim=1)
        embedding = torch.cat((embedding, action_vector.permute(1, 0)), dim=1)
        forward_prediction = self.forward_predictor(embedding).view(-1, 7, 7, 3)
        uncertainty = self.uncertainty_predictor(embedding).view(-1, 7, 7, 3)
        return forward_prediction, uncertainty
