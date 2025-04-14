import torch.nn as nn 

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Eerste 3x3 convolutie + batchnorm + activatie (met meegegeven stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Tweede 3x3 convolutie (stride altijd 1) + batchnorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (residuele verbinding)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Als dimensies niet overeenkomen: pas shortcut aan met 1x1 convolutie
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Hoofdpad
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Tel shortcut op bij output
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Eerste convolutielaag: 3-kanaals input (RGB), 64 filters
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Zelfgemaakte resnet-lagen volgens het klassieke schema:
        # (filters, aantal blocks, stride voor eerste block)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Adaptive average pooling reduceert feature map naar 1x1 per kanaal
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Volledig verbonden laag voor classificatie
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Genereert meerdere residual blocks in een laag
        # Alleen het eerste block gebruikt de meegegeven stride
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # update voor volgende block
        return nn.Sequential(*layers)

    def forward(self, x):
        # Inputverwerking: eerste conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Residuele blokken
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Pooling en flatten
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # Classificatie
        out = self.fc(out)
        return out
