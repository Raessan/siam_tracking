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
        """
        Converts an img to a tensor ready to be used in NN
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = (img[None].transpose(0,3,1,2) - mean) / std
        return torch.from_numpy(img[0])


def heatmap_center_of_mass(hm, p_mass=0.75):
    """
    Compute center-of-mass restricted to the top region
    that covers p_mass of the total heatmap probability.
    hm: 2D array (H,W)
    p_mass: float in (0,1) = fraction of prob mass to keep
    Returns (i_com, j_com) in heatmap coords (floats)
    """
    H, W = hm.shape
    total = hm.sum()
    if total <= 0:
        return ( (H-1)/2.0, (W-1)/2.0 )
    
    flat = hm.ravel().astype(np.float32)
    idx_sorted = np.argsort(flat)[::-1]  # descending
    cumsum = np.cumsum(flat[idx_sorted])
    cutoff = p_mass * total
    keep = idx_sorted[: np.searchsorted(cumsum, cutoff) + 1]

    rows, cols = np.unravel_index(keep, hm.shape)
    weights = flat[keep]
    weights /= weights.sum()

    i_com = (weights * rows).sum()
    j_com = (weights * cols).sum()
    return int(i_com), int(j_com)

def wh_from_regressor(hm, bbox_map, k=9):
    flat_idx = np.argpartition(hm.ravel(), -k)[-k:]   # indices of top-k (unordered)
    rows, cols = np.unravel_index(flat_idx, hm.shape)
    weights = hm[rows, cols].astype(np.float32)
    weights = weights / (weights.sum() + 1e-8)
    w = (weights * bbox_map[rows, cols, 0]).sum()
    h = (weights * bbox_map[rows, cols, 1]).sum()
    return w, h

# def wh_from_regressor(hm, bbox_map, p_mass=0.05):
#     """
#     Weighted average of regressed w,h over the region
#     that covers p_mass of the heatmap probability.
#     hm: (H,W), bbox_map: (H,W,2)
#     Returns (w,h) normalized (e.g. relative to search size)
#     """
#     total = hm.sum()
#     if total <= 0:
#         return bbox_map.mean(axis=(0,1))

#     flat = hm.ravel().astype(np.float32)
#     idx_sorted = np.argsort(flat)[::-1]
#     cumsum = np.cumsum(flat[idx_sorted])
#     cutoff = p_mass * total
#     keep = idx_sorted[: np.searchsorted(cumsum, cutoff) + 1]

#     rows, cols = np.unravel_index(keep, hm.shape)
#     weights = flat[keep] / flat[keep].sum()

#     w = (weights * bbox_map[rows, cols, 0]).sum()
#     h = (weights * bbox_map[rows, cols, 1]).sum()
#     return w, h