import cv2
import numpy as np
import os
import argparse
from math import sqrt
from constants import *

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Labeling"
CTRL_PT_RADIUS = 10


class Boundary:

    def __init__(self, width, height):
        self.top = height
        self.bottom = 0
        self.left = 0
        self.right = width

    def contains_point(self, x, y):
        has_vertical_boundary = (
            self.top is not None and self.bottom is not None)
        if has_vertical_boundary:
            return self.top > y and self.bottom < y

        has_horizontal_boundary = (
            self.left is not None and self.right is not None)
        if has_horizontal_boundary:
            return self.left < x and self.right > x

        return False

    def set_boundary_top(self, top):
        self.top = top

    def set_boundary_bottom(self, bottom):
        self.bottom = bottom

    def set_boundary_left(self, left):
        self.left = left

    def set_boundary_right(self, right):
        self.right = right


class ControlPoint:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = CTRL_PT_RADIUS
        self.in_use = True

    def draw(self, img):
        cv2.circle(img, (self.x, self.y), self.radius, BLUE, 1)

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def is_clicked(self, x, y):
        dist = sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        return dist <= self.radius

    def is_in_use(self):
        return self.in_use

    def destroy(self):
        self.in_use = False


class Marking:

    def __init__(self, point):
        self.points = [point]
        self.in_use = True

    def is_in_use(self):
        return self.in_use

    def add_point(self, point):
        self.points.append(point)

    def update(self):
        self.points = [point for point in self.points if point.is_in_use()]
        if len(self.points) == 0:
            self.in_use = False

    def draw(self, img, debug=True):
        color = RED if debug else WHITE
        pts = []
        for point in self.points:
            if debug:
                point.draw(img)
            pts.append([point.x, point.y])
        cv2.polylines(img, np.array([pts]), False, color)


class SceneMode:
    ADD_POINT = "ADD_POINT"
    ADD_MARKING = "ADD_MARKING"
    EDIT_POINT = "EDIT_POINT"
    DELETE_POINT = "DELETE_POINT"


class Scene:

    def __init__(self, width, height, frame_idx):
        self.frame_idx = frame_idx
        self.mode = SceneMode.ADD_MARKING
        self.points = []
        self.markings = []
        self.boundary = Boundary(width, height)
        self.active_point = None
        self.active_marking = None
        self.width = width
        self.height = height
        self.img = np.zeros((height, width, 3), np.uint8)

    def set_frame_index(self, frame_idx):
        self.frame_idx = frame_idx

    def update(self):
        self.img = np.zeros((self.height, self.width, 3), np.uint8)
        for marking in self.markings:
            marking.update()
        self.points = [point for point in self.points if point.is_in_use()]
        self.markings = [
            marking for marking in self.markings if marking.is_in_use()]

    def draw(self):
        cv2.putText(self.img, self.mode, (50, 50), FONT, 1, RED, 2)
        frame_info = "{} {}".format("Frame", self.frame_idx)
        cv2.putText(self.img, frame_info, (50, 100), FONT, 1, RED, 2)
        for marking in self.markings:
            marking.draw(self.img)

    def get_img(self):
        return self.img

    def save_img(self, filename):
        skeleton = np.zeros((self.height, self.width, 1), np.uint8)
        for marking in self.markings:
            marking.draw(skeleton, debug=False)
        cv2.imwrite(filename, skeleton)

    def change_mode(self, mode):
        if len(self.markings) == 0:
            self.mode = SceneMode.ADD_MARKING
        else:
            self.mode = mode

    def mouse_handle(self, event, x, y, flags, param):
        handle_functions = {
            cv2.EVENT_LBUTTONDOWN: self._on_lbutton_down,
            cv2.EVENT_LBUTTONUP: self._on_lbutton_up,
            cv2.EVENT_MOUSEMOVE: self._on_mouse_move
        }

        if self.boundary.contains_point(x, y):
            handle_functions.get(event, self._not_handle)(x, y)
        else:
            self.active_point = None

    def _on_lbutton_down(self, x, y):
        self._find_active_point(x, y)
        if self.active_point is None:
            if self.mode == SceneMode.ADD_POINT:
                new_point = ControlPoint(x, y)
                self.points.append(new_point)
                self.active_marking.add_point(new_point)
            if self.mode == SceneMode.ADD_MARKING:
                new_point = ControlPoint(x, y)
                new_marking = Marking(new_point)
                self.points.append(new_point)
                self.markings.append(new_marking)
                self.active_marking = new_marking
                self.mode = SceneMode.ADD_POINT
        else:
            if self.mode == SceneMode.DELETE_POINT:
                self.active_point.destroy()
                self.active_point = None

    def _find_active_point(self, x, y):
        for point in self.points:
            if point.is_clicked(x, y):
                self.active_point = point
                return

    def _on_lbutton_up(self, x, y):
        self.active_point = None

    def _on_mouse_move(self, x, y):
        if self.mode == SceneMode.EDIT_POINT and self.active_point is not None:
            self.active_point.move_to(x, y)

    def _not_handle(self, x, y):
        pass


def merge_images(img1, img2):
    fg_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fg_gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    fg = cv2.bitwise_and(img2, img2, mask=mask)
    return cv2.add(bg, fg)


def main(args):
    cap = cv2.VideoCapture()
    cap.open(args["video"])
    frame_idx = args["frame"]

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    basename = os.path.basename(args["video"])
    video_name, _ = os.path.splitext(basename)
    out_folder = os.path.join(GROUND_TRUTH_FOLDER, video_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WINDOW_NAME)
    scene = Scene(width, height, frame_idx)
    cv2.setMouseCallback(WINDOW_NAME, scene.mouse_handle)

    while frame_idx in range(0, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, img = cap.read()
        scene.set_frame_index(frame_idx)
        out_file = os.path.join(out_folder, str(frame_idx) + ".png")
        while True:
            scene.update()
            scene.draw()
            fg = scene.get_img()

            frame = merge_images(img, fg)
            cv2.imshow(WINDOW_NAME, frame)

            key = 0xFF & cv2.waitKey(1)
            if key == 27:
                exit()
            elif key == ord("1"):
                scene.change_mode(SceneMode.ADD_MARKING)
            elif key == ord("2"):
                scene.change_mode(SceneMode.ADD_POINT)
            elif key == ord("3"):
                scene.change_mode(SceneMode.DELETE_POINT)
            elif key == ord("4"):
                scene.change_mode(SceneMode.EDIT_POINT)
            elif key == ord("w"):
                frame_idx += 1
                break
            elif key == ord("q"):
                frame_idx -= 1
                break
            elif key == ord("e"):
                frame_idx += 1
                scene.save_img(out_file)
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to video")
    ap.add_argument("-f", "--frame", type=int, default=0,
                    help="(optional) start frame")
    main(vars(ap.parse_args()))
