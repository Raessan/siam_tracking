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