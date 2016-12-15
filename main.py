from post_processing import process_image, print_center
from utils import Timer
from constants import *
import sys
import os
import cv2
import model_handler
import argparse


if __name__ == "__main__":
    video_path = sys.argv[1]

    filename, extension = os.path.splitext(video_path)
    out_video_path = filename + "_output" + extension
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    with open(OUT_FILE, "w") as f:
        f.write("{0}\n".format(n_frames))

    m = model_handler.load_model("trained_models/carpet_model")

    timer = Timer()

    timer.start("Processing")
    for i in range(n_frames):
        _, img = cap.read()
        img, center = process_image(img, m, HALF_N_BLKS, WIDTH, HEIGHT, debug=True, flip=True)
        print_center(OUT_FILE, i, center)
        out.write(img)

        if 0xFF & cv2.waitKey(30) == 27:
            break

    timer.stop()
