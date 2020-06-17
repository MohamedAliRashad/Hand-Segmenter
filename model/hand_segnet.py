import cv2
import numpy as np
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


def maskIt(model, img, use_gpu=True):

    # device to run the model on
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    model.to(device)

    # convort BGR fromat to RGB
    image_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # compress input image for faster computation and less memory
    image_v = cv2.resize(image_raw, (320, 240))

    # normalize
    image_v = np.expand_dims((image_v.astype("float") / 255.0) - 0.5, 0)

    # rearrange dimensions to work with pytorch
    image_v = torch.from_numpy(image_v.transpose(0, 3, 1, 2)).float()

    # inference
    hand_scoremap_v = model(image_v.to(device))

    # rearrange output dimensions to be image like
    hand_scoremap_v = hand_scoremap_v.permute(0, 2, 3, 1).detach().cpu().numpy()

    # remove the extra dimension (batches dim)
    hand_scoremap_v = np.squeeze(hand_scoremap_v)

    # get the highest class prob for every pixel
    hand_scoremap_v = np.argmax(hand_scoremap_v, 2)

    return hand_scoremap_v


# Testing
if __name__ == "__main__":
    # Initialize model with trained weights
    weights_url = "https://drive.google.com/uc?export=download&id=1P_ZYUWuUjt2DAUtPxG882mdQX95TIX3D"
    seg = HandSegNet()
    seg.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))
    seg.eval()
    
    # # Dummy Test ...
    # input_size = (1, 3, 255, 255)
    # dummy = torch.randn(input_size)
    # print(seg(dummy).shape)

    # Live Demo ...
    cap = cv2.VideoCapture(0)
    key = " "
    while key != 113:
        ret, frame = cap.read()
        mask = maskIt(seg, frame)
        cv2.imshow("mask", mask.astype(np.float))
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
