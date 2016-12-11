from post_processing import process_image
from utils import Timer
import sys
import os
import cv2

OUT_FILE = "Contestant.txt"


def print_center(filename, index, center):
    with open(filename, "a") as f:
        line = "{0} {1} {2}\n".format(index, *center)
        f.write(line)

if __name__ == "__main__":
    video = sys.argv[1]
    filename, extension = os.path.splitext(video)
    out_video = filename + "_output" + extension

    cap = cv2.VideoCapture(video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    with open(OUT_FILE, "w") as f:
        f.write("{0}\n".format(n_frames))

    timer = Timer()

    timer.start("Processing")
    for i in range(n_frames):
        _, img = cap.read()
        img, center = process_image(img, 8, 272, 18)
        out.write(img)
        print_center(OUT_FILE, i, center)

    timer.stop()
