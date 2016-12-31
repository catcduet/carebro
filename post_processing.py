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
    predictions = [np.argmax(pred) for pred in predictions]
    return predictions


def calculate_out_map(out, predictions, offsets, width, height, margin):
    for i, offset in enumerate(offsets):
        patch = np.ones((height, width)) * predictions[i]
        out[offset[1]: offset[1] + height,
            offset[0]: offset[0] + width] += patch

    max_intensity = np.max(out)
    out = out / max_intensity * 255

    return out.astype(np.uint8)


def process_image(img, model, width, height, stride, margin):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    windows, offsets = get_windows(gray, width, height, stride, margin)
    predictions = predict(windows, model)
    out = np.zeros(gray.shape)
    out = calculate_out_map(out, predictions, offsets,
                            width, height, margin)

    return out


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
