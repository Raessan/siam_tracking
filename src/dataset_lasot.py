import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utils import get_context_bbox, crop_and_resize, to_tensor

class DatasetLaSOT(Dataset):
    def __init__(self, mode, dir_data, size_template, size_search, size_out, max_frame_sep, 
                 neg_prob=0.5, extra_context_template=0.5, min_extra_context_search=0.75, 
                 max_extra_context_search=1.0, max_shift=0, reg_full=False, prob_augment=0.5,
                 mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]):
        self.mode = mode
        self.dir_data = dir_data
        self.size_template = size_template
        self.size_search = size_search
        self.size_out = size_out
        self.neg_prob = neg_prob
        self.max_frame_sep = max_frame_sep
        self.extra_context_template = extra_context_template
        self.min_extra_context_search = min_extra_context_search
        self.max_extra_context_search = max_extra_context_search
        self.max_shift = max_shift
        self.reg_full = reg_full
        self.prob_augment = prob_augment
        # mean/std for ImageNet‐pretrained backbones
        # Adapt these variables to the backbone used
        self.mean = np.array(mean, dtype=np.float32)[None, :, None, None]
        self.std  = np.array(std, dtype=np.float32)[None, :, None, None]
        self.n_videos_train = 1000

        random.seed(42)

        if self.mode == "train" or self.mode == "val" or self.mode == "trainval":
            file = os.path.join(dir_data, "training_set.txt")
        elif self.mode == "test":
            file = os.path.join(dir_data, "testing_set.txt")
        else:
            raise Exception("the mode must be either train, val, trainval or test")

        # Get number of videos for the training set
        with open(file, 'r') as file:
            video_names = [line.strip() for line in file]

        shuffled = video_names[:]
        random.shuffle(shuffled)

        if self.mode == "train":
            self.video_names = shuffled[:self.n_videos_train]
        elif self.mode == "val":
            self.video_names = shuffled[self.n_videos_train:]
        else:
            self.video_names = shuffled

        # Get the category names
        self.categories = sorted([name for name in os.listdir(self.dir_data)
              if os.path.isdir(os.path.join(self.dir_data, name))])

        # Get the location of each frame per video and the bounding boxes (decided to keep as separate dictionaries
        self.dict_frames_per_video = {}
        self.dict_bboxes_per_video = {}
        for video_name in self.video_names:
            category = video_name.split('-')[0]
            valid_exts = {'.jpg', '.jpeg', '.png'}
            img_dir = os.path.join(dir_data, category, video_name, "img")
            self.dict_frames_per_video[video_name] = sorted([
                os.path.join(img_dir, frame)
                for frame in os.listdir(img_dir)
                if os.path.splitext(frame)[1].lower() in valid_exts
            ])
            #self.dict_frames_per_video[video_name] = sorted([os.path.join(dir_data, category, video_name, "img", frame) for frame in os.listdir(os.path.join(dir_data, category, video_name, "img"))])
            self.dict_bboxes_per_video[video_name] = []
            with open(os.path.join(dir_data, category, video_name, "groundtruth.txt"), "r") as f:
                for line in f:
                    bbox = list(map(int, line.strip().split(",")))
                    self.dict_bboxes_per_video[video_name].append(bbox)


        # Get the number of frames per video
        self.dict_n_frames_per_video = {}
        for key, value in self.dict_frames_per_video.items():
            self.dict_n_frames_per_video[key] = len(value)

        # Total number of frames
        self.total_n_frames = sum(self.dict_n_frames_per_video.values())

        # List of frames
        #self.list_frames = []
        #for key in sorted(self.dict_frames_per_video.keys()):
            #self.list_frames.extend(self.dict_frames_per_video[key])

    def get_data_from_idx(self, idx):
        cumulative = 0
        for key in sorted(self.dict_n_frames_per_video):
            value = self.dict_n_frames_per_video[key]
            if idx < cumulative + value:
                return key, idx-cumulative
            cumulative += value
        raise IndexError("Index out of range")

    def __len__(self):
        return self.total_n_frames
        
    
    def visualize_video(self, video_name, with_bboxes=True, fps=30):
        frame_delay_ms = int(1000/fps)
        for n_frame, frame_path in enumerate(self.dict_frames_per_video[video_name]):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Could not read {frame_path}")
                continue

            if with_bboxes:
                x, y, w, h = self.dict_bboxes_per_video[video_name][n_frame]
                # Draw rectangle: image, top-left, bottom-right, color (BGR), thickness
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            cv2.imshow("Video Playback", frame)

            if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def preprocess_pair(self, frame1, frame2, bbox1, bbox2):
        """
        Given two frames and their tight bboxes, compute exemplar & search patches,
        along with their resized bbox coordinates in patch space.
        Returns:
          exemplar_img, search_img, exemplar_box, search_box
        where boxes are (x, y, w, h) in the resized patch coordinate system.
        
        """
        extra_context_search = random.uniform(self.min_extra_context_search, self.max_extra_context_search)
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)
        
        # Frame 1: exemplar
        cx1, cy1, size1 = get_context_bbox(bbox1, self.extra_context_template)
        exemplar, scale1 = crop_and_resize(frame1, cx1, cy1, size1, self.size_template, 0, 0)
    
        # bbox1 in exemplar coords: centered
        ex_bbox = [ (self.size_template - bbox1[2]*scale1)/2,
                    (self.size_template - bbox1[3]*scale1)/2,
                    bbox1[2]*scale1,
                    bbox1[3]*scale1 ]
    
        # Frame 2: search
        cx2, cy2, size2 = get_context_bbox(bbox2, extra_context_search)
        search, scale2 = crop_and_resize(frame2, cx2, cy2, size2, self.size_search, shift_x, shift_y)
    
        # bbox2 in search coords before augment: centered
        sr_bbox = [ (self.size_search - bbox2[2]*scale2)/2 - shift_x,
                    (self.size_search - bbox2[3]*scale2)/2 - shift_y,
                    bbox2[2]*scale2,
                    bbox2[3]*scale2 ]
    
        return exemplar, search, ex_bbox, sr_bbox
    
    def get_random_bbox(self, image_shape, min_size_ratio=0.1, max_size_ratio=0.5):
        """
        Generate a random bounding box within an image.

        Args:
            image_shape: tuple (H, W)
            min_size_ratio: minimum size of box as a ratio of image
            max_size_ratio: maximum size of box as a ratio of image

        Returns:
            [x, y, w, h]
        """
        H, W = image_shape[:2]

        # Random box width and height
        w = random.randint(int(min_size_ratio * W), int(max_size_ratio * W))
        h = random.randint(int(min_size_ratio * H), int(max_size_ratio * H))

        # Ensure the box fits in the image
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        return [x, y, w, h]

    def get_negative_sample(self, video_name_first_frame):
        # Decide if the negative sample will have an object or pure background
        if random.random() < 0.5:
            with_object = True
        else:
            with_object = False
        # Select a video to sample from
        video_name_second_frame = video_name_first_frame
        while video_name_second_frame == video_name_first_frame:
            video_name_second_frame = random.choice(list(self.dict_n_frames_per_video.keys()))
        # Random frame
        idx_second_frame = random.randint(0, self.dict_n_frames_per_video[video_name_second_frame] -1)
        second_frame = cv2.imread(self.dict_frames_per_video[video_name_second_frame][idx_second_frame])
        # Get the bounding box of the object
        bbox2 = self.dict_bboxes_per_video[video_name_second_frame][idx_second_frame]
        # Check that bbox2 is valid and otherwise create a random bbox
        random_bbox = False
        if bbox2[2] == 0 or bbox2[3] == 0:
            # Create a random bbox to patch over it
            bbox2 = self.get_random_bbox(second_frame.shape)
            random_bbox = True
        if with_object or random_bbox == True: # Make sure that it has an object
            # We return directly the new frame and Bounding box
            return second_frame, bbox2, video_name_second_frame
        else: # Get a random patch of the image. For that, we are going to move randomly the bbox center and apply the same logic
            w, h = bbox2[2], bbox2[3]
            # Make sure we are within the limits
            max_x = second_frame.shape[1] - w
            max_y = second_frame.shape[0] - h
            x = random.randint(0,max_x)
            y = random.randint(0,max_y)
            return second_frame, [x, y, w, h], video_name_second_frame

    def make_rect_tent(self, bbox):
        """
        bbox = [xmin, ymin, w, h] in search-patch pixel coords.
        Returns heatmap of shape (H,W), peak=1 at center,
        linearly decaying to 0 at the box edges.
        """
        stride = self.size_search / self.size_out
    
        # 1) cell centers in pixel coords
        coords = np.arange(self.size_out) * stride + stride/2
        xs, ys = np.meshgrid(coords, coords)  # (H,W)
    
        xmin, ymin, w, h = bbox
        # Clamp values
        xmin = np.clip(xmin, 0, self.size_search)
        ymin = np.clip(ymin, 0, self.size_search)
        w = np.clip(w, 0, self.size_search-xmin)
        h = np.clip(h, 0, self.size_search-ymin)
        
        # Get centers
        cx = xmin + w/2
        cy = ymin + h/2
    
        # 2) normalized distances in [0,1]
        dx = np.abs(xs - cx) / (w/2)
        dy = np.abs(ys - cy) / (h/2)
    
        # 3) clamp and compute tent
        tx = np.clip(1 - dx, 0, 1)
        ty = np.clip(1 - dy, 0, 1)
        heatmap = tx * ty  # (H,W)
        # Force max to 1
        heatmap /= (heatmap.max()+1e-8)

        # 4) build (w,h) regression map
        mask = (heatmap > 0).astype(np.float32)      # (H,W)
        
        # 4) width/height normalized
        w_norm = (w  / self.size_search) * mask
        h_norm = (h  / self.size_search) * mask

        if self.reg_full:
            # 5) center offsets normalized
            dx_off = ((xs - cx) / self.size_search) * mask    # can be in [-0.5,0.5]
            dy_off = ((ys - cy) / self.size_search) * mask

            # 6) stack regressors: [dx, dy, w, h]
            reg = np.stack([dx_off, dy_off, w_norm, h_norm], axis=-1)  # (H,W,4)
        
        else:
            reg = np.stack([w_norm, h_norm], axis=-1)
        # reg_wh = np.zeros((self.size_out, self.size_out, 2), dtype=np.float32)
        # reg_wh[..., 0] = w/self.size_search * mask  # normalized width
        # reg_wh[..., 1] = h/self.size_search * mask  # normalized height

        return heatmap, reg

    def get_positive_sample(self, video_name, idx_first_frame):
        # Obtain the second idx and image
        idx_second_frame = idx_first_frame
        min_frame = max(0, idx_first_frame-self.max_frame_sep)
        max_frame = min(self.dict_n_frames_per_video[video_name], idx_first_frame+self.max_frame_sep)
        while idx_second_frame == idx_first_frame:
            #idx_second_frame = random.choice(range(self.dict_n_frames_per_video[video_name]))
            idx_second_frame = random.choice(range(min_frame, max_frame))
            
        second_frame = cv2.imread(self.dict_frames_per_video[video_name][idx_second_frame])

        # Obtain bounding boxes
        bbox2 = self.dict_bboxes_per_video[video_name][idx_second_frame]
        return second_frame, bbox2

    #def get_output(self, search_img, search_bbox):

    def photometric_augment(self, img,
                        brightness_delta=32,
                        contrast_range=(0.8,1.2),
                        saturation_range=(0.8,1.2),
                        hue_delta=10,
                        p_jitter=0.5,
                        p_blur=0.3,
                        p_noise=0.3):
        """
        img: H×W×3 BGR uint8
        Returns: same shape uint8
        """
        out = img.astype(np.float32)

        # 1) brightness
        if random.random() < p_jitter:
            delta = random.uniform(-brightness_delta, brightness_delta)
            out += delta

        # 2) contrast
        if random.random() < p_jitter:
            alpha = random.uniform(*contrast_range)
            out *= alpha

        # 3) saturation & hue (convert to HSV in OpenCV)
        if random.random() < p_jitter:
            hsv = cv2.cvtColor(out.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] *= random.uniform(*saturation_range)  # sat
            hsv[...,0] += random.uniform(-hue_delta, hue_delta)  # hue
            out = cv2.cvtColor(hsv.clip(0,255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 4) Gaussian blur
        if random.random() < p_blur:
            k = random.choice([3,5])  # odd kernel size
            out = cv2.GaussianBlur(out, (k,k), 0)

        # 5) Additive Gaussian noise
        if random.random() < p_noise:
            sigma = random.uniform(5, 20)
            noise = np.random.randn(*out.shape) * sigma
            out += noise

        return out.clip(0,255).astype(np.uint8)
        
            
    def __getitem__(self, idx):
        """ Returns the inputs and output for the learning problem. The input
        consists of an reference image tensor and a search image tensor, the
        output is the corresponding label tensor.

        Args:
            idx: (int) The index of a sequence inside the whole dataset, from
                which the function will choose the reference and search frames.

        Returns:
            ref_frame (torch.Tensor): The reference frame with the
                specified size.
            srch_frame (torch.Tensor): The search frame with the
                specified size.
            label (torch.Tensor): The label created with the specified
                function in self.label_fcn.
        """
        # Negative sample
        if random.random() < self.neg_prob:
            is_positive = False
        else: 
            is_positive = True
        
        # Obtain the video and the index inside that video. Then, read the imae
        video_name, idx_first_frame = self.get_data_from_idx(idx)

        # Obtain bounding boxes
        try:
            bbox1 = self.dict_bboxes_per_video[video_name][idx_first_frame]
        except:
            print("idx: ", idx)
            print("video name: ", video_name)
            print("idx_first_frame: ", idx_first_frame)

        # If the frame does not have the object, grab another random image
        while bbox1[2]==0 or bbox1[3]==0:
            idx_first_frame = random.randint(0, self.dict_n_frames_per_video[video_name] -1)
            # Obtain bounding boxes
            bbox1 = self.dict_bboxes_per_video[video_name][idx_first_frame]

        first_frame = cv2.imread(self.dict_frames_per_video[video_name][idx_first_frame])

        if is_positive: # Positive sample
            second_frame, bbox2 = self.get_positive_sample(video_name, idx_first_frame)
            video_search_name = video_name
            # If the width or height of the box are 0 it is actually a negative sample!!
            if bbox2[2]==0 or bbox2[3]==0:
                is_positive = False
                # Create a random bbox to patch over it
                bbox2 = self.get_random_bbox(second_frame.shape)

        else: # Negative sample
            second_frame, bbox2, video_search_name = self.get_negative_sample(video_name)

        template, search, bbox1_x1y1wh, bbox2_x1y1wh = self.preprocess_pair(first_frame, second_frame, bbox1, bbox2)

        if is_positive:
            heatmap, reg_wh = self.make_rect_tent(bbox2_x1y1wh)
        else:
            if self.reg_full:
                reg_wh = np.zeros((self.size_out, self.size_out, 4), dtype=np.float32)
            else:  
                reg_wh = np.zeros((self.size_out, self.size_out, 2), dtype=np.float32)
            heatmap = np.zeros((self.size_out, self.size_out), dtype=np.float32)

        if random.random() < self.prob_augment:
            template = self.photometric_augment(template)
            search = self.photometric_augment(search)
        return to_tensor(template, self.mean, self.std), to_tensor(search, self.mean, self.std), heatmap, reg_wh, video_name, video_search_name

if __name__ == "__main__":
    dataset = DatasetLaSOT("val", "/home/rafa/deep_learning/datasets/LaSOT", 127, 255, 25, 10, 0.45, 0.5, 0.75, 1.5, 32, False, False)
    output = dataset.__getitem__(10000)
    dataset.visualize_video("umbrella-8")