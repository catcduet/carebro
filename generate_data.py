import cv2
import numpy as np
import argparse
import os
import uuid
from constants import *
from utils import Timer


def get_label(window, threshold):
    n_points = np.count_nonzero(window)
    return n_points >= threshold


class Margin:

    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right


def generate_data(img, truth, width, height, stride,
                  threshold, margin, pos_folder, neg_folder):
    x_offset = 0
    y_offset = 0
    left = margin.left
    right = img.shape[1] - margin.right
    bottom = img.shape[0] - margin.bottom
    top = margin.top
    img = img[top: bottom, left: right]
    truth = truth[top: bottom, left: right]
    img_height, img_width = img.shape[:2]
    while (x_offset + width <= img_width):
        while (y_offset + height <= img_height):
            img_window = img[y_offset: y_offset + height,
                             x_offset: x_offset + width]
            truth_window = truth[y_offset: y_offset + height,
                                 x_offset: x_offset + width]
            label = get_label(truth_window, threshold)

            filename = str(uuid.uuid4()).replace("-", "") + ".png"

            if label:
                out_file = os.path.join(pos_folder, filename)
            else:
                out_file = os.path.join(neg_folder, filename)
            cv2.imwrite(out_file, img_window)
            y_offset += stride
        y_offset = 0
        x_offset += stride


def main(args):
    cap = cv2.VideoCapture()
    cap.open(args["video"])

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    width = args["width"]
    height = args["height"]
    stride = args["stride"]
    threshold = args["threshold"]
    margin = Margin(384, 0, 88, 88)

    basename = os.path.basename(args["video"])
    video_name, _ = os.path.splitext(basename)
    truth_folder = os.path.join(GROUND_TRUTH_FOLDER, video_name)

    pos_folder = os.path.join(DATA_FOLDER, video_name, "1")
    neg_folder = os.path.join(DATA_FOLDER, video_name, "0")
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)
    if not os.path.exists(neg_folder):
        os.makedirs(neg_folder)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timer = Timer()
    timer.start("generating data")
    for i in range(n_frames):
        print("{}/{}".format(i + 1, n_frames))
        _, img = cap.read()
        truth_path = os.path.join(truth_folder, str(i) + ".png")
        truth = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE)
        generate_data(img, truth, width, height,
                      stride, threshold, margin, pos_folder, neg_folder)
    timer.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("video", type=str, help="path to video")
    ap.add_argument("-h", "--height", required=True, type=int,
                    help="window height")
    ap.add_argument("-w", "--width", required=True, type=int,
                    help="window width")
    ap.add_argument("-s", "--stride", required=True, type=int,
                    help="window stride")
    ap.add_argument("-t", "--threshold", required=True, type=int,
                    help="number of points to consider a window is positive")
    ap.print_help()
    main(vars(ap.parse_args()))
