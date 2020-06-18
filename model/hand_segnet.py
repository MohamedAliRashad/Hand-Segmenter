import torch
import torch.nn as nn
import torch.nn.functional as F


class HandSegNet(nn.Module):
    def __init__(self, num_classes=2, out_size=None):
        super(HandSegNet, self).__init__()
        self.out_size = out_size

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.layer_6 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        if self.out_size is None:
            out = F.upsample(out, size=(x.shape[2], x.shape[3]), mode="bilinear")
        else:
            out = F.upsample(
                out, size=(self.out_size[0], self.out_size[1]), mode="bilinear"
            )

        return out


def handsegnet(pretrained=False, progress=True, **kwargs):
    """
    Load `HandSegNet` model architecture 

    Args:
    ---
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = HandSegNet(**kwargs)
    weights_url = "https://drive.google.com/uc?export=download&id=1P_ZYUWuUjt2DAUtPxG882mdQX95TIX3D"
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(weights_url, progress=progress)
        model.load_state_dict(state_dict)
    return model


# Testing
if __name__ == "__main__":

    # dummy test to make sure everything works
    seg = handsegnet()
    input_size = (1, 3, 255, 255)
    dummy = torch.randn(input_size)
    print(seg(dummy).shape)
