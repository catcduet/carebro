from post_processing import process_image
from utils import Timer, Margin
from constants import *
import os
import cv2
import model_handler
import argparse


def main(args):
    video_path = args["video"]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    filename, extension = os.path.splitext(video_path)
    out_video_path = filename + "_output" + extension
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    with open(OUT_FILE, "w") as f:
        f.write("{0}\n".format(n_frames))

    m = model_handler.load_model("trained_models/video01_model")

    timer = Timer()

    margin = Margin(288, 0, 88, 88)
    stride = args["stride"]

    for i in range(n_frames):
        timer.start("Processing frame {}".format(str(i)))
        _, img = cap.read()
        out = process_image(img, m, WIDTH, HEIGHT, stride, margin)

        cv2.imshow("Image", img)
        cv2.imshow("Out", out)

        timer.stop()

        cv2.waitKey(0)
        if 0xFF & cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to video")
    ap.add_argument("-s", "--stride", type=int, help="window stride")
    main(vars(ap.parse_args()))
