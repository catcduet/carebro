import cv2
import numpy as np


def split_image(img, half_n_blks, blk_width, blk_height, flip):
    blks = []
    coords = []

    img_height, img_width = img.shape[:2]

    x0 = int(img_width / 2 - blk_width)
    x1 = x0 + blk_width
    x2 = x1 + blk_width
    y0 = img_height - blk_height
    y1 = img_height

    for i in range(half_n_blks):
        # left block
        blks.append(img[y0:y1, x0:x1])
        coords.append((x0, y0 + int(blk_height / 2)))

        if flip:
            # right block will be flipped horizontally
            flipped_blk = cv2.flip(img[y0:y1, x1:x2], 1)
            blks.append(flipped_blk)
            coords.append((x2, y0 + int(blk_height / 2)))
        else:
            blks.append(img[y0:y1, x1:x2])
            coords.append((x1, y0 + int(blk_height / 2)))

        y0 -= blk_height
        y1 -= blk_height

    return blks, coords


def predict_lane_markings(blks, model):
    n_blks = len(blks)
    blks = np.array(blks)
    blks = blks.reshape(n_blks, blks.shape[1], blks.shape[2], 1).astype('float32')

    labels = model.predict(blks, batch_size=n_blks, verbose=0)

    labels = [np.argmax(label) for label in labels]
    print(labels)
    return labels


def get_lane_marking_points(coords, labels, flip):
    left_pts = []
    right_pts = []

    for i, label in enumerate(labels):
        if label == 0:
            continue

        point = (coords[i][0] + label - 1, coords[i][1])

        if i % 2 == 0:
            left_pts.append(point)
        else:
            if flip:
                point = (coords[i][0] - label + 1, coords[i][1])
            right_pts.append(point)

    return left_pts, right_pts


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


def process_image(img, model, half_n_blks, blk_width, blk_height, debug=False, flip=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blks, coords = split_image(gray, half_n_blks, blk_width, blk_height, flip)
    labels = predict_lane_markings(blks, model)
    left_pts, right_pts = get_lane_marking_points(coords, labels, flip)

    upper_bound = img.shape[0]
    lower_bound = upper_bound - blk_height * half_n_blks

    left_pt0, left_pt1 = fit_line(left_pts, lower_bound, upper_bound)
    right_pt0, right_pt1 = fit_line(right_pts, lower_bound, upper_bound)

    if (left_pt0 and left_pt1 and right_pt0 and right_pt1):
        center = get_lane_center(left_pt0, left_pt1, right_pt0, right_pt1)
        cv2.circle(img, center, 4, (255, 0, 0), 3)
        cv2.line(img, left_pt0, left_pt1, (0, 0, 255), 2)
        cv2.line(img, right_pt0, right_pt1, (0, 0, 255), 2)
    else:
        center = (0, 0)

    if debug:
        dot_color = (0, 255, 255)
        padding = 10
        bottom = half_n_blks * (blk_height + padding)

        for pt in left_pts:
            cv2.circle(img, pt, 2, dot_color, -1)
        for pt in right_pts:
            cv2.circle(img, pt, 2, dot_color, -1)

        for i, blk in enumerate(blks):
            colored_blk = cv2.cvtColor(blk, cv2.COLOR_GRAY2BGR)
            cv2.putText(colored_blk, str(labels[i]), (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            if labels[i] != 0:
                marking_pos = (labels[i] - 1, int(blk_height / 2))
                cv2.circle(colored_blk, marking_pos, 2, dot_color, -1)

            x_pos = 0 if i % 2 == 0 else blk_width
            y_pos = bottom - int(i / 2) * (blk_height + padding)

            cv2.imshow(str(i), colored_blk)
            cv2.moveWindow(str(i), x_pos, y_pos)

        cv2.imshow("Frame", img)

    return img, center


def print_center(filename, index, center):
    with open(filename, "a") as f:
        line = "{0} {1} {2}\n".format(index, *center)
        f.write(line)


if __name__ == "__main__":
    img = cv2.imread("D:/lena.jpg")
    flip = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.imshow("Flip", flip)
    cv2.waitKey(0)
