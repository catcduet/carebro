from post_processing import process_image, non_maxima_suppression, print_center
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

    gmean = 0  # global mean
    count_gmean = 0
    recent_centers = []  # some number of recent center points
    default_center = (333, 504)  # center of the video
    threshold_distance = 100  # pixels to be considered noisy (not movement)

    for i in range(n_frames):
        timer.start("Processing frame {}".format(str(i)))
        _, img = cap.read()
        out, raw_center = process_image(img, m, WIDTH, HEIGHT, stride, margin)
        
        # Perform center tracking
        mean = None
        center = None
        if len(recent_centers) == 0:
            if raw_center[0] == 0:
                center = default_center
                # don't add to recent_centers so this "fake" center won't effect prediction
            else:
                recent_centers.append(raw_center[0])  # first real center
                center = raw_center
                gmean = raw_center[0]
                count_gmean += 1
        else:
            mean = np.mean(recent_centers)
            # measurement is unknown or too vary (can't be movement)
            if raw_center[0] == 0 or abs(raw_center[0] - mean) > threshold_distance:
                center = (int(mean), default_center[1])  # use prediction (mean)
                #  mean doesn't change because using prediction is somewhat "fake"
                continue

            elif gmean != 0 and abs(mean - gmean) > threshold_distance:  # too vary, can't be movement
                center = (int(gmean), default_center[1])  # use global mean
            else:  # having both measurement and prediction
                center = (int((mean + raw_center[0]) / 2), default_center[1])
            
            # update the recent_centers list
            if len(recent_centers) == 5:
                # remove element that varies the most from the current center point
                recent_centers.pop(np.argmax([abs(k - center[0]) for k in recent_centers]))
            recent_centers.append(center[0])
            
            # update global mean
            gmean = (gmean * count_gmean + center[0]) / (count_gmean + 1)
            count_gmean += 1

        # debugging
        # print(center[0], raw_center[0])
        # print(recent_centers)
        # print(mean, gmean)

        cv2.circle(img, raw_center, 4, RED, 2)
        cv2.circle(img, center, 4, GREEN, 2)
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
