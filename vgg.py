import torch.nn as nn 

class VGG_16(nn.Module): 
    def __init__(self, num_classes=1000):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 4 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=7*7*512, out_features=4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5), 

            nn.Linear(in_features=4096, out_features=4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5), 

            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x): 
        x = self.feature_extractor(x) 
        x = self.classifier(x) 
        return x 