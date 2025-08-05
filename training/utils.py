import cv2
import numpy as np
import torch

def get_context_bbox(bbox, extra_context):
        """
        Convert a tight bbox (x, y, w, h) into a square context bbox with margin.
        Returns (cx, cy, size).
        """
        x, y, w, h = bbox
        cx = x + w / 2.
        cy = y + h / 2.
        # context padding = (w+h)*extra_context
        pad = (w + h) * extra_context
        # square size
        size = np.sqrt((w + pad) * (h + pad))
        return cx, cy, size

def crop_and_resize(frame,
                cx, cy,
                size,
                out_size,
                shift_x=0, shift_y=0):
    """
    Crop a square patch of side 'size' centered at (cx, cy) from frame,
    apply a shift in the out_size coordinate system, pad with border
    replication if needed, and resize to (out_size, out_size).

    shift_x, shift_y: pixel offsets **in the resized patch**. Can be
    positive or negative, moving the target around in the crop.
    Returns the patch and the scaling factor.
    """
    h, w = frame.shape[:2]

    # 1) convert shifts from output coords → original-frame coords
    #    scale = out_size / size  ⇒  size/out_size = 1/scale
    shift_x_orig = shift_x * size / out_size
    shift_y_orig = shift_y * size / out_size

    # 2) adjust the true crop-center in the original frame
    cx = cx + shift_x_orig
    cy = cy + shift_y_orig

    # 3) now compute the square-window coords as before
    x1 = cx - size/2
    y1 = cy - size/2
    x2 = x1 + size
    y2 = y1 + size

    # 4) compute padding amounts for out‑of‑bounds regions
    left   = int(max(0, -np.floor(x1)))
    top    = int(max(0, -np.floor(y1)))
    right  = int(max(0, np.ceil(x2)  - w))
    bottom = int(max(0, np.ceil(y2)  - h))

    # 5) pad & crop
    padded = cv2.copyMakeBorder(
        frame,
        top, bottom, left, right,
        borderType=cv2.BORDER_REPLICATE
    )
    x1p, y1p = x1 + left,  y1 + top
    x2p, y2p = x2 + left,  y2 + top
    patch    = padded[int(y1p):int(y2p), int(x1p):int(x2p)]

    # 6) resize & return
    patch_resized = cv2.resize(patch, (out_size, out_size))
    scale = out_size / size
    return patch_resized, scale

def to_tensor(img, mean, std):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = (img[None].transpose(0,3,1,2) - mean) / std
        return torch.from_numpy(img[0])

def x1y1wh_cxcywh(x1, y1, w, h):

    cx = int(x1 + w/ 2.0)
    cy = int(y1 + h/ 2.0)
    return cx, cy, w, h

def cxcywh_x1y1wh(cx, cy, w, h):

    x1 = int(cx - w/ 2.0)
    y1 = int(cy - h/ 2.0)
    return x1, y1, w, h

def cxcywh_x1y1x2y2(cx, cy, w, h):

    x1 = int(cx - w/ 2.0)
    y1 = int(cy - h/ 2.0)
    x2 = int(cx + w/ 2.0)
    y2 = int(cy + h/ 2.0)
    return x1, y1, x2, y2

def x1y1wh_x1y1x2y2(x1, y1, w, h):

    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def bbox_center_to_relative(cx, cy, w, h, img_w, img_h):

    cx_rel = cx / img_w
    cy_rel = cy / img_h
    w_rel = w / img_w
    h_rel = h / img_h
    return cx_rel, cy_rel, w_rel, h_rel

def heatmap_center_of_mass(hm):
    """
    hm:  2D array of shape (H, W)
    Returns:
      (i_com, j_com): center of mass in heatmap‐grid coords (floats)
    """
    H, W = hm.shape
    total = hm.sum()
    if total == 0:
        # no mass → fall back to the exact center of the grid
        return ( (H-1)/2.0, (W-1)/2.0 )
    # build index grids
    i = np.arange(H)[:, None]   # shape (H,1)
    j = np.arange(W)[None, :]   # shape (1,W)
    # weighted sum of indices
    i_com = (i * hm).sum() / total
    j_com = (j * hm).sum() / total
    return int(i_com), int(j_com)