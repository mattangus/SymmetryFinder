import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from fastdtw import fastdtw
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

def get_score(img):
    img = img[...,::-1] #Convert BGR to RGB order

    im_R = img[...,0]
    im_G = img[...,1]
    im_B = img[...,2]

    m1 = im_R < 155
    m2 = im_B > 210
    mask = np.logical_and(m1, m2)

    mask = mask.astype(np.uint8)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # get mask contours
    cnts = cv2.findContours(255 - opened,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2]

    #get largest contour
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    # do polygon approx
    eps = 0.0005
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    points = approx[:,0].copy()
    # y axis is flipped in image space
    points[:,1] = img.shape[0] - points[:,1]

    max_x, max_y = np.max(points, axis=0)
    min_x, min_y = np.min(points, axis=0)

    mid_x = (max_x + min_x)/2

    points_centered = points - [mid_x, 0]
    flipped = points_centered.copy()
    flipped[:,0] = -flipped[:,0]

    errors = []
    for i in range(len(flipped)):
        start = flipped[i]
        end = flipped[(i + 1) % len(flipped)]
        errors.append(np.linalg.norm(points_centered[0] - start) +
                    np.linalg.norm(np.linalg.norm(points_centered[-1] - end)))

    # plt.plot(errors)
    min_i = np.argmin(errors) + 1
    flipped = np.flip(np.concatenate([flipped[min_i:], flipped[:min_i]]), 0)

    def get_grad(X,Y):
        # D can also be an arbitrary distance matrix: numpy array, shape [m, n]
        D = SquaredEuclidean(X,Y)
        sdtw = SoftDTW(D, gamma=1.0)
        # soft-DTW discrepancy, approaches DTW as gamma -> 0
        value = sdtw.compute()
        # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
        E = sdtw.grad()
        # gradient w.r.t. X, shape = [m, d]
        G = D.jacobian_product(E)
        mean_grad = np.mean(G,0)
        return mean_grad, value

    lr = 0.01
    losses = []
    for i in range(250):
        grad, loss = get_grad(flipped, points_centered)
        losses.append(loss)
        flipped = flipped - [grad[0]*lr,0] #only use x

    poly1 = Polygon(points_centered)
    poly2 = Polygon(flipped)

    inter_poly = poly1.intersection(poly2)
    # union_poly = poly1.union(poly2)

    inter = inter_poly.area
    union = poly1.area + poly2.area - inter

    iou = inter/union
    return iou


vid = cv2.VideoCapture("imgs/example.mp4")

is_good, img = vid.read()
i = 0

ious = []
while is_good:
    iou = get_score(img)
    ious.append(iou)
    print(i, ":", iou)

    is_good, img = vid.read()
    i += 1

max_i = np.argmax(ious)
print("best:", max_i, ious[max_i])

min_i = np.argmin(ious)
print("worst:", min_i, ious[min_i])

plt.plot(ious)
plt.show()