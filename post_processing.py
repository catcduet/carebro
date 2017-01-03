import cv2
import numpy as np
from constants import *


def get_windows(img, width, height, stride, margin):
    windows = []
    offsets = []

    left = margin.left
    right = img.shape[1] - margin.right
    bottom = img.shape[0] - margin.bottom
    top = margin.top

    x_offset = left
    y_offset = top

    while (x_offset + width <= right):
        while (y_offset + height <= bottom):
            img_window = img[y_offset: y_offset + height,
                             x_offset: x_offset + width]

            img_window = img_window.flatten() / 255.0
            windows.append(img_window)
            offsets.append((x_offset, y_offset))

            y_offset += stride
        y_offset = top
        x_offset += stride

    return windows, offsets


def predict(windows, model):
    n_windows = len(windows)
    windows = np.array(windows)
    windows = windows.reshape(n_windows, HEIGHT,
                              WIDTH, 1).astype('float32')
    predictions = model.predict(windows, batch_size=128)

    # returns 1 if window is marking, -1 otherwise
    # predictions = [2 * np.argmax(pred) - 1 for pred in predictions]

    # returns (marking probability) - (non-marking probability)
    predictions = [pred[1] - pred[0] for pred in predictions]

    return predictions


def get_split_point(heat_map, margin):
    # TODO: somehow split 2 lane markings
    return 360


def get_points(heat_map, margin):
    left_points = []
    right_points = []

    left = margin.left
    right = heat_map.shape[1] - margin.right
    bottom = heat_map.shape[0] - margin.bottom
    top = margin.top

    split_point = get_split_point(heat_map, margin)

    r = 2
    sz = 2 * r + 1
    z = sz * sz * 255

    integral = cv2.integral(heat_map)

    for x in range(left, right, sz):
        for y in range(top, bottom, sz):
            x1 = min(x + sz, bottom)
            y1 = min(y + sz, bottom)
            cx = int((x1 + x) / 2)
            cy = int((y1 + y) / 2)
            maybe_point = (integral[y][x] + integral[y1][x1] -
                           integral[y][x1] - integral[y1][x]) / z

            maybe_point = heat_map[y][x] / 255
            rand = np.random.random() * 0
            if maybe_point > rand:
                if x > split_point:
                    right_points.append([cx, cy])
                else:
                    left_points.append([cx, cy])
    return np.array(left_points), np.array(right_points)


def calculate_heat_map(heat_map, predictions, offsets, width, height, margin):
    for i, offset in enumerate(offsets):
        patch = np.ones((height, width)) * predictions[i]
        heat_map[offset[1]: offset[1] + height,
                 offset[0]: offset[0] + width] += patch

    # clips heat map to [0, 255]
    max_intensity = np.max(heat_map)
    heat_map = heat_map / max_intensity * 255 * (heat_map > 0)

    return heat_map.astype(np.uint8)


def fit_line(points, y0, y1):
    if len(points) < 1:
        return None, None

    vx, vy, cx, cy = cv2.fitLine(np.float32(
        points), cv2.DIST_HUBER, 0, 0.01, 0.01)

    if vy == 0:
        return None, None

    x0 = (y0 - cy) * vx / vy + cx
    x1 = (y1 - cy) * vx / vy + cx
    return (x0, y0), (x1, y1)


def get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1):
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


def process_image(img, model, width, height, stride, margin, debug=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    windows, offsets = get_windows(gray, width, height, stride, margin)
    predictions = predict(windows, model)
    heat_map = np.zeros(gray.shape)
    heat_map = calculate_heat_map(heat_map, predictions, offsets,
                                  width, height, margin)
    _, thresh = cv2.threshold(
        heat_map, HEAT_MAP_THRESH, 255, cv2.THRESH_TOZERO)

    left_points, right_points = get_points(thresh, margin)

    upper_bound = img.shape[0]
    lower_bound = margin.top

    left_pt0, left_pt1 = fit_line(left_points, lower_bound, upper_bound)
    right_pt0, right_pt1 = fit_line(right_points, lower_bound, upper_bound)

    # get lane's center
    if (left_pt0 and left_pt1 and right_pt0 and right_pt1):
        center = get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1)
    else:
        center = (0, 504)  # y is fixed

    if debug:
        x = get_split_point(heat_map, margin)
        y = margin.top

        debug = np.zeros((*gray.shape, 3))
        cv2.line(debug, (x, y), (x, 576), WHITE)
        cv2.line(debug, (0, y), (720, y), WHITE)

        for point in left_points:
            cv2.circle(debug, tuple(point), 1, BLUE)
        for point in right_points:
            cv2.circle(debug, tuple(point), 1, RED)

        if left_pt0 is not None and left_pt1 is not None:
            cv2.line(debug, left_pt0, left_pt1, GREEN)

        if right_pt0 is not None and right_pt1 is not None:
            cv2.line(debug, right_pt0, right_pt1, GREEN)

        cv2.imshow("Heat Map", heat_map)
        cv2.imshow("Debug", debug)

    if left_pt0 is not None and left_pt1 is not None:
        cv2.line(img, left_pt0, left_pt1, RED)

    if right_pt0 is not None and right_pt1 is not None:
        cv2.line(img, right_pt0, right_pt1, RED)

    return img, center, left_points, right_points


def non_maxima_suppression(src, sz):
    """
    Ported from this code: http://code.opencv.org/attachments/994/nms.cpp
    """
    M, N = src.shape[:2]
    block = np.ones((2 * sz + 1, 2 * sz + 1), np.uint8) * 255
    dst = np.zeros((M, N), np.uint8)

    for m in range(0, M, sz + 1):
        for n in range(0, N, sz + 1):
            # get the maximal candidate within the block
            ic_start = m
            ic_end = min(m + sz + 1, M)
            ic_size = ic_end - ic_start

            jc_start = n
            jc_end = min(n + sz + 1, N)
            jc_size = jc_end - jc_start

            patch = src[ic_start: ic_end, jc_start: jc_end]
            _, vcmax, _, ijmax = cv2.minMaxLoc(patch)
            cc = tuple(map(sum, zip(ijmax, (jc_start, ic_start))))

            # search the neighbours centered around the candidate for the true
            # maxima
            in_start = max(cc[1] - sz, 0)
            in_end = min(cc[1] + sz + 1, M)
            in_size = in_end - in_start

            jn_start = max(cc[0] - sz, 0)
            jn_end = min(cc[0] + sz + 1, N)
            jn_size = jn_end - jn_start

            # mask out the block whose maxima we already know
            blockmask = block[0: in_size, 0: jn_size].copy()

            iis_start = ic_start - in_start
            iis_end = min(ic_start - in_start + sz + 1, in_size)
            iis_size = iis_end - iis_start

            jis_start = jc_start - jn_start
            jis_end = min(jc_start - jn_start + sz + 1, jn_size)
            jis_size = jis_end - jis_start

            blockmask[iis_start: iis_end, jis_start: jis_end] = np.zeros(
                (iis_size, jis_size), np.uint8)

            patch = src[in_start: in_end, jn_start: jn_end]
            _, vnmax, _, _ = cv2.minMaxLoc(patch, blockmask)

            # loose condition
            if vcmax >= vnmax and vcmax != 0:
                dst[cc[1], cc[0]] = 255

    return dst


if __name__ == "__main__":
    patch = np.ones((5, 5), np.uint8)
    patch = patch * 10 * 0.8
    patch = patch.astype(np.uint8)
    print(patch)
