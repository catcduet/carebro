import cv2
import numpy as np
from constants import *


class ROI:

    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.initialize()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def initialize(self):
        self.vx = 0
        self.vy = 1
        self.cx = self.left
        self.cy = int((self.bottom - self.top) / 2)
        self.width = self.right - self.left
        self.dirty = True

    def set_attributes(self, cx, width):
        self.cx = cx
        self.width = width

    def update(self, line, width):
        if line is None:
            return False, 0

        vx, vy, cx, cy = line
        if vy == 0:
            angle = np.arccos(self.vx * vx)
            return False, angle

        # calculate angle between two vectors of two frames
        # cosin = self.vx * vx + self.vy * vy
        # angle = np.arccos(cosin)

        # calculate angle between Oy
        angle = np.arccos(vy)

        # angle smaller than 90 degree
        absolute_angle = np.arccos(abs(vy))

        print("angle (line, Oy): ", angle * 180 / PI,
              "abs_angle: ", absolute_angle * 180 / PI)

        if absolute_angle > ANGLE_THRESH:    # noise
            return False, angle

        self.width = int(abs(width / vy))
        self.vx = vx
        self.vy = vy
        x0 = int((self.cy - cy) * vx / vy + cx)
        self.cx = x0 - int(self.width / 2)
        self.dirty = False
        return True, angle

    def get_center_line(self):
        line = (self.vx, self.vy, self.cx + int(self.width / 2), self.cy)
        return line

    def get_x_range(self, y, width, stride):
        x = int((y - self.cy) * self.vx / self.vy + self.cx)
        stop = min(x + self.width - width, self.right)
        start = max(x, self.left)
        return range(start, stop, stride)

    def get_y_range(self, height, stride):
        return range(self.top, self.bottom - height, stride)

    def debug(self, img, color):
        left_line = (self.vx, self.vy, self.cx, self.cy)
        right_line = (self.vx, self.vy, self.cx + self.width, self.cy)
        left_pt1, left_pt2 = get_endpoints(left_line, self.top, self.bottom)
        right_pt1, right_pt2 = get_endpoints(right_line, self.top, self.bottom)
        cv2.line(img, left_pt1, left_pt2, color)
        cv2.line(img, right_pt1, right_pt2, color)


class PostProcessing:

    def __init__(self, model, width, height, stride, margin, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.model = model
        self.width = width
        self.height = height
        self.stride = stride
        self.left = margin.left
        self.right = self.img_w - margin.right
        self.bottom = self.img_h - margin.bottom
        self.top = margin.top
        self.left_roi = ROI(self.left, self.right, self.top, self.bottom)
        self.right_roi = ROI(self.left, self.right, self.top, self.bottom)
        self.saved_left_line = None
        self.saved_right_line = None

    def process(self, img, debug=True):
        if debug:
            debug_img = np.zeros((self.img_h, self.img_w, 3))
            self.left_roi.debug(debug_img, BLUE)
            self.right_roi.debug(debug_img, RED)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows, offsets = self._get_windows(gray)
        predictions = self._predict(windows)
        heat_map = self._calculate_heat_map(predictions, offsets)
        _, thresh = cv2.threshold(
            heat_map, HEAT_MAP_THRESH, 255, cv2.THRESH_TOZERO)
        left_points, right_points = self._get_points(thresh)

        left_line = fit_line(left_points)
        right_line = fit_line(right_points)

        dirty = False
        if left_line is None and right_line is None:
            # no lane markings
            center = (0, 0)
            dirty = True
        else:
            # update region of interest
            left_ok, angle = self.left_roi.update(left_line, ROI_WIDTH)
            right_ok, angle = self.right_roi.update(right_line, ROI_WIDTH)

            print(left_ok, right_ok)

            # use saved left line and saved right line if not ok
            if not left_ok:
                dirty = True
                left_line = (self.saved_left_line
                             if self.saved_left_line else left_line)
            elif not self.left_roi.dirty:
                self.saved_left_line = left_line

            if not right_ok:
                dirty = True
                right_line = (self.saved_right_line
                              if self.saved_right_line else right_line)
            elif not self.right_roi.dirty:
                self.saved_right_line = right_line

            left_pt0, left_pt1 = get_endpoints(
                left_line, self.top, self.bottom)
            right_pt0, right_pt1 = get_endpoints(
                right_line, self.top, self.bottom)

            center = get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1)

            cv2.line(img, left_pt0, left_pt1, RED)
            cv2.line(img, right_pt0, right_pt1, RED)

            if debug:
                cv2.line(debug_img, left_pt0, left_pt1, GREEN)
                cv2.line(debug_img, right_pt0, right_pt1, GREEN)

        if dirty:
            print("[WARNING] Dirty => Reset ROI")
            self.left_roi.initialize()
            self.right_roi.initialize()

        if debug:
            x = self._get_split_point(thresh)
            y = self.top

            cv2.line(debug_img, (x, y), (x, 576), WHITE)
            cv2.line(debug_img, (0, y), (720, y), WHITE)

            for point in left_points:
                cv2.circle(debug_img, tuple(point), 1, BLUE)
            for point in right_points:
                cv2.circle(debug_img, tuple(point), 1, RED)

            cv2.imshow("Heat Map", thresh)
            cv2.imshow("Debug", debug_img)

        return img, center, left_points, right_points

    def _get_windows(self, img):
        windows = []
        offsets = []

        # if left_roi == right_roi then processing in left_roi only
        ignore_right_roi = False
        if self.left_roi == self.right_roi:
            ignore_right_roi = True

        def append_windows_and_offsets(roi):
            for y_offset in roi.get_y_range(self.height, self.stride):
                for x_offset in roi.get_x_range(y_offset,
                                                self.width, self.stride):
                    img_window = img[y_offset: y_offset + self.height,
                                     x_offset: x_offset + self.width]
                    img_window = img_window.flatten() / 255.0
                    windows.append(img_window)
                    offsets.append((x_offset, y_offset))

        append_windows_and_offsets(self.left_roi)

        if not ignore_right_roi:
            append_windows_and_offsets(self.right_roi)

        return windows, offsets

    def _predict(self, windows):
        n_windows = len(windows)
        windows = np.array(windows)
        windows = windows.reshape(n_windows, HEIGHT,
                                  WIDTH, 1).astype('float32')
        predictions = self.model.predict(windows, batch_size=128)

        # returns 1 if window is marking, -1 otherwise
        # predictions = [2 * np.argmax(pred) - 1 for pred in predictions]

        # returns (marking probability) - (non-marking probability)
        predictions = [pred[1] - pred[0] for pred in predictions]

        return predictions

    def _calculate_heat_map(self, predictions, offsets):
        heat_map = np.zeros((self.img_h, self.img_w))
        for i, offset in enumerate(offsets):
            patch = np.ones((self.height, self.width)) * predictions[i]
            heat_map[offset[1]: offset[1] + self.height,
                     offset[0]: offset[0] + self.width] += patch

        # clips heat map's value range to [0, 255]
        max_intensity = np.max(heat_map)
        heat_map = heat_map / max_intensity * 255 * (heat_map > 0)

        return heat_map.astype(np.uint8)

    def _get_points(self, heat_map):
        left_points = []
        right_points = []

        lroi = self.left_roi
        rroi = self.right_roi

        # update left_roi and right_roi using split point if they're the same
        if self.left_roi == self.right_roi:
            split_point = self._get_split_point(heat_map)
            lroi_width = split_point - self.left
            lroi.set_attributes(lroi.cx, lroi_width)
            rroi_width = self.right - split_point
            rroi.set_attributes(split_point, rroi_width)

        # relates to density of points
        r = POINT_DENSITY
        sz = 2 * r + 1
        z = sz * sz * 255

        integral = cv2.integral(heat_map)

        # for each window (sz x sz) in roi, if window has enough white pixels
        # then considering center of window is a point
        def append_points(points, roi):
            y_range = roi.get_y_range(sz, sz)
            for y in y_range:
                x_range = roi.get_x_range(y, sz, sz)
                for x in x_range:
                    x1 = min(x + sz, x_range.stop)
                    y1 = min(y + sz, y_range.stop)
                    cx = int((x1 + x) / 2)
                    cy = int((y1 + y) / 2)
                    maybe_point = (integral[y][x] + integral[y1][x1] -
                                   integral[y][x1] - integral[y1][x]) / z

                    # maybe_point = heat_map[y][x] / 255
                    threshold = POINT_THRESH
                    if maybe_point > threshold:
                        points.append([cx, cy])

        append_points(left_points, lroi)
        append_points(right_points, rroi)

        return np.array(left_points), np.array(right_points)

    def _get_split_point(self, heat_map):
        padding = 40
        stride = 5
        lin = int((self.left + self.right) / 2)
        rin = lin
        lout = lin + padding
        rout = rin - padding
        integral = cv2.integral(heat_map)
        l = self.left
        r = self.right
        b = self.bottom
        t = self.top

        sum_lin = (integral[t][l] + integral[b][lin + 1] -
                   integral[t][lin + 1] - integral[b][l])
        sum_lout = (integral[t][l] + integral[b][lout + 1] -
                    integral[t][lout + 1] - integral[b][l])
        while sum_lout > sum_lin:
            lin += stride
            lout += stride
            sum_lin = (integral[t][l] + integral[b][lin + 1] -
                       integral[t][lin + 1] - integral[b][l])
            sum_lout = (integral[t][l] + integral[b][lout + 1] -
                        integral[t][lout + 1] - integral[b][l])

        sum_rin = (integral[t][rin] + integral[b][r] -
                   integral[t][r] - integral[b][rin])
        sum_rout = (integral[t][rout] + integral[b][r] -
                    integral[t][r] - integral[b][rout])
        while sum_rout > sum_rin:
            rin -= stride
            rout -= stride
            sum_rin = (integral[t][rin] + integral[b][r] -
                       integral[t][r] - integral[b][rin])
            sum_rout = (integral[t][rout] + integral[b][r] -
                        integral[t][r] - integral[b][rout])

        split_point = lin if 0 < sum_lin < sum_rin else rin
        return split_point


def get_endpoints(line, y0, y1):
    vx, vy, cx, cy = line
    if vy == 0:
        return None, None

    x0 = int((y0 - cy) * vx / vy + cx)
    x1 = int((y1 - cy) * vx / vy + cx)
    return (x0, y0), (x1, y1)


def fit_line(points):
    if len(points) < 1:
        return None

    vx, vy, cx, cy = cv2.fitLine(np.float32(
        points), cv2.DIST_HUBER, 0, 0.01, 0.01)

    return (vx, vy, cx, cy)


def get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1):
    if (left_pt0 is None or left_pt1 is None or
            right_pt0 is None or right_pt1 is None):
        return (0, 0)
    m_x = (left_pt0[0] + left_pt1[0]) / 2
    m_y = (left_pt0[1] + left_pt1[1]) / 2
    m_u = (right_pt0[0] + right_pt1[0]) / 2
    m_v = (right_pt0[1] + right_pt1[1]) / 2

    cx = (m_x + m_u) / 2
    cy = (m_y + m_v) / 2

    cx = int(cx)
    cy = int(cy)

    return (cx, cy)


def print_center(filename, index, center):
    with open(filename, "a") as f:
        line = "{0} {1} {2}\n".format(index, *center)
        f.write(line)


def center_tracking(img, recent_centers, gmean, count_gmean, raw_center,
                    left_points, right_points, threshold_distance=100,
                    trusted_num_points=100, debug=True):
    # threshold_distance: pixels to be considered noisy (not movement)
    # trusted_num_points: number of points to be considered trustful
    # recent_centers: some number of recent center points
    # gmean: global mean
    # count_gmean
    default_center = (333, 504)  # center of the video

    # Perform center tracking
    mean = None
    center = None
    if len(recent_centers) == 0:
        if raw_center[0] == 0:
            center = default_center
            # don't add to recent_centers so this "fake" center won't effect
            # prediction
        else:
            recent_centers.append(raw_center[0])  # first real center
            center = raw_center
            gmean = raw_center[0]
            count_gmean += 1
    else:
        mean = np.mean(recent_centers)
        # measurement is unknown or too vary (can't be movement)
        if (raw_center[0] == 0 or
                abs(raw_center[0] - mean) > threshold_distance):
            center = (int(mean), default_center[1])  # use prediction (mean)
            #  mean doesn't change because using prediction is somewhat "fake"
            update_mean = False

        # too vary, can't be movement
        elif gmean != 0 and abs(mean - gmean) > threshold_distance:
            center = (int(gmean), default_center[1])  # use global mean
            update_mean = True

        else:  # having both measurement and prediction
            alpha = 0.5  # trust degree of measurement
            # ratio between 2 point sets' length
            len_left = len(left_points)
            len_right = len(right_points)
            if len_left == 0 or len_right == 0:
                a = 0
            else:
                a = len_left / len_right if len_left < len_right else len_right / len_left
            b = len_left / trusted_num_points
            c = len_right / trusted_num_points
            score = (b + c) * a
            alpha = score if score <= 1 else 1
            center = (int((1 - alpha) * mean + alpha *
                          raw_center[0]), default_center[1])
            update_mean = True

        if update_mean:
            # update the recent_centers list
            if len(recent_centers) == 5:
                # remove element that varies the most from the current center
                # point
                recent_centers.pop(
                    np.argmax([abs(k - center[0]) for k in recent_centers]))
            recent_centers.append(center[0])

            # update global mean
            gmean = (gmean * count_gmean + center[0]) / (count_gmean + 1)
            count_gmean += 1

    # show centers if debugging
    if debug:
        cv2.circle(img, raw_center, 4, RED, 2)
        cv2.circle(img, center, 4, GREEN, 2)
    return center


if __name__ == "__main__":
    patch = np.ones((5, 5), np.uint8)
    patch = patch * 10 * 0.8
    patch = patch.astype(np.uint8)
    print(patch)
