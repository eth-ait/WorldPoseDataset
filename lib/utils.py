import numpy as np


def project_points(obj_pts, R, t, f, principal_points, k, *args, **kwargs):
    # world to camera transformation
    pts_c = obj_pts @ R.T + t
    img_pts = pts_c[..., :2] / pts_c[..., 2:]

    # here we add a small hack to make sure the points are in the image
    r = np.square(img_pts).sum(-1, keepdims=True)

    # we assume the distortion will only have minor impact on the projection
    r = np.clip(r, 0, 0.5 / min(max(np.abs(k).max(), 1), 1))

    # if k.shape[0] <= 2:
    #     img_pts = img_pts * (1 + k[0] * r + k[1] * np.square(r))
    # else:
    #     img_pts = img_pts * (1 + k[0] * r + k[1] * np.square(r) + k[2] * np.power(r, 3))
    d = np.ones_like(r)
    for i in range(0, k.shape[0]):
        d = d + k[i] * np.power(r, i + 1)
    img_pts = img_pts * d
    img_pts = img_pts * f + principal_points[None, :]
    return img_pts