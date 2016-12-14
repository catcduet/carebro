import cv2
import numpy as np

import model_handler

m = model_handler.load_model("deeplane_model")


def split_image(img, n_blks, blk_width, blk_height):
    blks = []
    coords = []

    img_height, img_width = img.shape[:2]

    x0 = int(img_width / 2 - blk_width)
    x1 = x0 + blk_width
    x2 = x1 + blk_width
    y0 = img_height - blk_height
    y1 = img_height

    for i in range(n_blks):
        # left block
        blks.append(img[y0:y1, x0:x1])
        coords.append((x0, y0 + int(blk_height / 2)))

        # right block
        blks.append(img[y0:y1, x1:x2])
        coords.append((x1, y0 + int(blk_height / 2)))

        y0 -= blk_height
        y1 -= blk_height

    return blks, coords


def predict_lane_markings(blks):
    blks = np.array(blks)
    #print(blks.shape)
    blks = blks.reshape(16, 1, blks.shape[1], blks.shape[2]).astype('float32')
    #print(blks.shape)
    labels = m.predict(blks, batch_size=16, verbose=0)

    labels = [np.argmax(label) for label in labels]

    return labels


def get_lane_marking_points(coords, labels):
    left_pts = []
    right_pts = []

    for i, label in enumerate(labels):
        if label == 0:
            continue

        point = (coords[i][0] + label - 1, coords[i][1])
        if i % 2 == 0:
            left_pts.append(point)
        else:
            right_pts.append(point)

    return left_pts, right_pts


def fit_line(points, y0, y1):
    if len(points) < 1:
        return None
    vx, vy, cx, cy = cv2.fitLine(np.float32(
        points), cv2.DIST_L2, 0, 0.01, 0.01)
    x0 = (y0 - cy) * vx / vy + cx
    x1 = (y1 - cy) * vx / vy + cx
    return (x0, y0), (x1, y1)


def get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1):
    m_x = (left_pt0[0] + left_pt1[0]) / 2
    m_y = (left_pt0[1] + left_pt1[1]) / 2
    m_u = (right_pt0[0] + right_pt1[0]) / 2
    m_v = (right_pt0[1] + right_pt1[1]) / 2

    cx = int((m_x + m_u) / 2)
    cy = int((m_y + m_v) / 2)

    return (cx, cy)


def process_image(img, n_blks, blk_width, blk_height):
    blks, coords = split_image(img, n_blks, blk_width, blk_height)
    labels = predict_lane_markings(blks)
    left_pts, right_pts = get_lane_marking_points(coords, labels)

    upper_bound = img.shape[0]
    lower_bound = upper_bound - blk_height * n_blks
    left_pt0, left_pt1 = fit_line(left_pts, lower_bound, upper_bound)
    right_pt0, right_pt1 = fit_line(right_pts, lower_bound, upper_bound)
    center = get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1)

    cv2.circle(img, center, 4, (255, 0, 0), 3)
    cv2.line(img, left_pt0, left_pt1, (0, 0, 255), 2)
    cv2.line(img, right_pt0, right_pt1, (0, 0, 255), 2)

    return img, center

if __name__ == "__main__":
    cap = cv2.VideoCapture()
    cap.open("../video/01.avi")

    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blks, coords = split_image(img, 8, 272, 18)
    labels = predict_lane_markings(blks)
    left_pts, right_pts = get_lane_marking_points(coords, labels)

    for pt in left_pts:
        cv2.circle(img, pt, 2, (0, 0, 255), -1)
    for pt in right_pts:
        cv2.circle(img, pt, 2, (0, 0, 255), -1)

    left_pt0, left_pt1 = fit_line(left_pts, 432, 576)
    right_pt0, right_pt1 = fit_line(right_pts, 432, 576)

    center = get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1)
    cv2.circle(img, center, 4, (0, 0, 0), 3)

    cv2.line(img, left_pt0, left_pt1, (0, 0, 255))
    cv2.line(img, right_pt0, right_pt1, (0, 0, 255))

    cv2.imshow("Image", img)
    for i in range(16):
        cv2.imshow(str(i), blks[i])

    cv2.waitKey(0)
