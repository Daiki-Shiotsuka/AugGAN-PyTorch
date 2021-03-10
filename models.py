import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channnels=3, out_channels=64):
        super(Encoder, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 3
        self.out_channels = out_channels # out_channels = 64

        layers = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channnels, out_channels, kernel_size=7, padding=0),
        nn.BatchNorm2d(out_channels)
        ]

        in_channnels, out_channels = out_channels, out_channels*2 # in_channnels = 64, out_channels = 128
        for i in range(2):
            layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channnels, out_channels, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            ]
            in_channnels, out_channels = out_channels, out_channels*2

        # in_channnels = 256
        for i in range(3):
            layers += [
            ResBlock(in_channnels)
            ]

        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channnels=256, out_channels=3):
        super(Generator, self).__init__()

        self.in_channnels = in_channnels # in_channnels = 256
        self.out_channels = out_channels # out_channels = 3 (RGB)
        middle_channels = int(in_channnels/4) # middle_channels = 64

        self.hard = HardShare(in_channnels)
        self.soft = SoftShare(in_channnels)
        self.decG = Decoder_Generator(middle_channels, out_channels)

    def forward(self, x):
        return self.decG(self.soft(self.hard(x)))


class ParsingNetworks(nn.Module):
    def __init__(self, in_channnels=256, seg_channels=20):
        super(ParsingNetworks, self).__init__()

        self.in_channnels = in_channnels # in_channnels = 256
        self.seg_channels = seg_channels # seg_channels = segmentation class number
        middle_channels = int(in_channnels/4) # middle_channels = 64

        self.hard = HardShare(in_channnels)
        self.soft = SoftShare(in_channnels)
        self.decS = Decoder_ParsingNetworks(middle_channels, seg_channels)
    def forward(self, x):
        return self.decS(self.soft(self.hard(x)))


class Discriminator(nn.Module):
    def __init__(self, in_channnels=3, out_channels=64):
        super(Discriminator, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 3 (RGB)
        self.out_channels = out_channels # out_channels = 64

        layers = []
        for i in range(4):
            layers += [
            nn.Conv2d(in_channnels, out_channels, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
            ]
            in_channnels, out_channels = out_channels, out_channels*2

        #in_channnels, out_channels = 512, 1024
        layers += [
        nn.Conv2d(in_channnels, in_channnels, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(in_channnels),
        nn.LeakyReLU(0.2, True)
        ]

        layers += [
        nn.Conv2d(in_channnels, 1, kernel_size=4, stride=1, padding=2)
        #nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class HardShare(nn.Module):
    def __init__(self, in_channnels=256):
        super(HardShare, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 256

        layers = []
        for i in range(6):
            layers += [
            ResBlock(in_channnels)
            ]

        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class SoftShare(nn.Module):
    def __init__(self, in_channnels=256):
        super(SoftShare, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 256

        layers = []
        for i in range(2):
            layers += [
            nn.ConvTranspose2d(in_channnels, int(in_channnels/2), kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(int(in_channnels/2)),
            nn.ReLU(inplace=True)
            ]
            in_channnels = int(in_channnels/2)

        # in_channnels = 64
        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class Decoder_Generator(nn.Module):
    def __init__(self, in_channnels=64, out_channels=3):
        super(Decoder_Generator, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 64
        self.out_channels = out_channels # out_channels = 3

        layers = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channnels, out_channels, kernel_size=7, padding=0),
        nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class Decoder_ParsingNetworks(nn.Module):
    def __init__(self, in_channnels=64, seg_channels=20):
        super(Decoder_ParsingNetworks, self).__init__()
        self.in_channnels = in_channnels # in_channnels = 64
        self.seg_channels = seg_channels # out_channels = num of segmentation class

        layers = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channnels, int(in_channnels/2), kernel_size=7, padding=0),
        nn.BatchNorm2d(int(in_channnels/2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_channnels/2), seg_channels, kernel_size=1)
        #nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, channnels):
        super(ResBlock, self).__init__()
        layers = [
        nn.ReflectionPad2d(1),
        nn.Conv2d(channnels, channnels, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(channnels),
        nn.ELU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(channnels, channnels, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(channnels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input_data):
        return input_data + self.model(input_data)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
