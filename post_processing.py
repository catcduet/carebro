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
        patch = np.ones((height, width))
        patch = (patch * predictions[i])
        out[offset[1]: offset[1] + height,
            offset[0]: offset[0] + width] += patch

    return out


def process_image(img, model, width, height, stride, margin):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    windows, offsets = get_windows(gray, width, height, stride, margin)
    predictions = predict(windows, model)
    out = np.zeros(gray.shape)
    out = calculate_out_map(out, predictions, offsets,
                            width, height, margin)
    max_intensity = np.max(out)
    out = out / max_intensity * 255
    return out.astype(np.uint8)


if __name__ == "__main__":
    patch = np.ones((5, 5), np.uint8)
    patch = patch * 10 * 0.8
    patch = patch.astype(np.uint8)
    print(patch)
