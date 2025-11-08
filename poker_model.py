import torch.nn as nn
import torch.nn.functional as F

# Model Architecture Constants
INPUT_FEATURES = 119
LAYER_1_OUT_FEATURES = 128
LAYER_2_OUT_FEATURES = 64
LAYER_3_OUT_FEATURES = 16
OUTPUT_FEATURES = 1

class PokerModel(nn.Module):
    """A feed-forward neural network to predict the equity of post-flop poker hands.
    
    The model takes a 119 feature vector representing the hand and board state
    and outputs a raw logit representing the hand's predicted equity.
    """
    def __init__(self):                    
        super().__init__()
        self.hidden_layer1 = nn.Linear(INPUT_FEATURES, LAYER_1_OUT_FEATURES)
        self.batch_norm1 = nn.BatchNorm1d(LAYER_1_OUT_FEATURES)

        self.hidden_layer2 = nn.Linear(LAYER_1_OUT_FEATURES, LAYER_2_OUT_FEATURES)
        self.batch_norm2 = nn.BatchNorm1d(LAYER_2_OUT_FEATURES)

        self.hidden_layer3 = nn.Linear(LAYER_2_OUT_FEATURES, LAYER_3_OUT_FEATURES)
        self.batch_norm3 = nn.BatchNorm1d(LAYER_3_OUT_FEATURES)

        self.output_layer = nn.Linear(LAYER_3_OUT_FEATURES, OUTPUT_FEATURES)       

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.batch_norm1(x)
        x = F.silu(x)

        x = self.hidden_layer2(x)
        x = self.batch_norm2(x)
        x = F.silu(x)

        x = self.hidden_layer3(x)
        x = self.batch_norm3(x)
        x = F.silu(x)

        return self.output_layer(x)
    
