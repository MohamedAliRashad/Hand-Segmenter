import argparse
from model.hand_segnet import handsegnet
import torch
import cv2
import numpy as np


def main(use_gpu=True):

    # import model
    model = handsegnet(pretrained=True)

    # device to run the model on
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    model.to(device)

    # Live Demo ...
    cap = cv2.VideoCapture(0)
    key = " "

    while key != 113:
        # read frames and return state
        ret, frame = cap.read()

        # convort BGR fromat to RGB
        image_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        # display original frame and mask
        cv2.imshow("mask", hand_scoremap_v.astype(np.float))
        cv2.imshow("frame", frame)

        # wait for 'q' to stop
        key = cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add Arguments needed
    parser.add_argument(
        "--no-gpu", action="store_false", default=True, help="disables CUDA inference"
    )

    args = parser.parse_args()

    main(use_gpu=args.no_gpu)
