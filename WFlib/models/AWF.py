import torch
from torch import nn

class AWF(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.feature_extraction = nn.Sequential(
            nn.Dropout(p=0.25),

            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            # ✅ make output length fixed (independent of seq_len)
            nn.AdaptiveAvgPool1d(45),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 45, num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x)
        return x