import numpy as np
import cv2

from utils import *

class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox_x1y1x2y2, out_sz, padding=(0, 0, 0)):
        bbox_x1y1x2y2 = [float(x) for x in bbox_x1y1x2y2]
        a = (out_sz-1) / (bbox_x1y1x2y2[2]-bbox_x1y1x2y2[0])
        b = (out_sz-1) / (bbox_x1y1x2y2[3]-bbox_x1y1x2y2[1])
        c = -a * bbox_x1y1x2y2[0]
        d = -b * bbox_x1y1x2y2[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox_x1y1x2y2, crop_bbox_cxcywh, size):
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        # if self.scale:
        #     scale_x = (1.0 + Augmentation.random() * self.scale)
        #     scale_y = (1.0 + Augmentation.random() * self.scale)
        #     w, h = crop_bbox_cxcywh[2], crop_bbox_cxcywh[3]
        #     scale_x = min(scale_x, float(im_w) / w)
        #     scale_y = min(scale_y, float(im_h) / h)
        #     crop_bbox_cxcywh[2] *= scale_x
        #     crop_bbox_cxcywh[3] *= scale_y
            

        crop_bbox_x1y1x2y2 = cxcywh_x1y1x2y2(*crop_bbox_cxcywh)
        # if self.shift:
        #     sx = Augmentation.random() * self.shift
        #     sy = Augmentation.random() * self.shift

        #     x1, y1, x2, y2 = crop_bbox_x1y1x2y2

        #     sx = max(-x1, min(im_w - 1 - x2, sx))
        #     sy = max(-y1, min(im_h - 1 - y2, sy))

        #     crop_bbox_x1y1x2y2 = [x1 + sx, y1 + sy, x2 + sx, y2 + sy]

        # adjust target bounding box
        x1, y1 = crop_bbox_x1y1x2y2[0], crop_bbox_x1y1x2y2[1]
        bbox_x1y1x2y2 = bbox_x1y1x2y2[0] -x1, bbox_x1y1x2y2[1] -y1, bbox_x1y1x2y2[2] -x1, bbox_x1y1x2y2[3] -y1

        # if self.scale:
        #     bbox_x1y1x2y2 = [bbox_x1y1x2y2[0]/scale_x, bbox_x1y1x2y2[1]/scale_y, bbox_x1y1x2y2[2]/scale_x, bbox_x1y1x2y2[3]/scale_y]

        image = self._crop_roi(image, crop_bbox_x1y1x2y2, size)
        return image, bbox_x1y1x2y2

    def _flip_aug(self, image, bbox_x1y1x2y2):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = width - 1 - bbox_x1y1x2y2[2], bbox_x1y1x2y2[1], width -1 - bbox_x1y1x2y2[0], bbox_x1y1x2y2[3]
        return image, bbox

    def __call__(self, image, bbox_x1y1x2y2, size, gray=False):
        shape = image.shape
        crop_bbox_cxcywh = [shape[0]//2, shape[1]//2, size-1, size-1]
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox_x1y1x2y2 = self._shift_scale_aug(image, bbox_x1y1x2y2, crop_bbox_cxcywh, size)
        return image, bbox_x1y1x2y2

        # # color augmentation
        # if self.color > np.random.random():
        #     image = self._color_aug(image)

        # # blur augmentation
        # if self.blur > np.random.random():
        #     image = self._blur_aug(image)

        # # flip augmentation
        # if self.flip and self.flip > np.random.random():
        #     image, bbox_x1y1x2y2 = self._flip_aug(image, bbox_x1y1x2y2)
        # return image, bbox_x1y1x2y2

        image = self._crop_roi(image, bbox_x1y1x2y2, size)
        return image, bbox_x1y1x2y2