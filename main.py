from post_processing import PostProcessing, center_tracking, print_center
from utils import Timer, Margin
from constants import *
import os
import cv2
import model_handler
import argparse
import numpy as np


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
    out_video = cv2.VideoWriter(
        out_video_path, fourcc, fps, (width, height), isColor=False)

    with open(OUT_FILE, "w") as f:
        f.write("{0}\n".format(n_frames))

    m = model_handler.load_model("trained_models/12345_100k_30_5_13")

    timer = Timer()

    margin = Margin(432, 0, 88, 88)
    stride = args["stride"]

    gmean = 0
    count_gmean = 0
    recent_centers = []

    pp = PostProcessing(m, WIDTH, HEIGHT, stride, margin, width, height)

    for i in range(n_frames):
        timer.start("Processing frame {}".format(str(i)))
        _, img = cap.read()

        out, raw_center, left_points, right_points = pp.process(img)

        center = center_tracking(img, recent_centers, gmean, count_gmean,
                                 raw_center, left_points, right_points)

        print_center(OUT_FILE, i, center)

        cv2.imshow("Output", out)

        out_video.write(out)

        timer.stop()

        cv2.waitKey(0)
        if 0xFF & cv2.waitKey(5) == 27:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to video")
    ap.add_argument("-s", "--stride", type=int,
                    default=5, help="window stride")
    main(vars(ap.parse_args()))
