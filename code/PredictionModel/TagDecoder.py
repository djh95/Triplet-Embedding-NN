import torch.nn as nn

class TagDecoder(nn.Module):
    def __init__(self, Feature_Dimensions, tag_number, dropout_probability=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(Feature_Dimensions, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(1024, tag_number),
            nn.Sigmoid()
        )
 
 
    def forward(self, x):
        """
        :param [b, feature_dimensionality]:
        :return [b, tag_number]:
        """
        # decoder
        x = self.decoder(x)
        return x