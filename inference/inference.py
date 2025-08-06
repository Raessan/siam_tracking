import torch
import cv2
import numpy as np
from src.model import SiameseTracker
from src.utils import get_context_bbox, crop_and_resize, to_tensor, heatmap_center_of_mass
import config.config as cfg

# Globals to track ROI
drawing = False
p1 = p2 = None
roi_defined = False
current_mouse_pos = (0, 0)
perform_inference = False

def on_mouse(event, x, y, flags, param):
    global drawing, p1, p2, roi_defined, current_mouse_pos
    current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        p2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        p2 = (x, y)
        drawing = False
        roi_defined = True

def main():

    MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
    EXTRA_CONTENT = cfg.EXTRA_CONTEXT_TEMPLATE
    SIZE_SEARCH = cfg.SIZE_SEARCH
    SIZE_TEMPLATE = cfg.SIZE_TEMPLATE
    SIZE_OUT = cfg.SIZE_OUT
    REG_FULL = cfg.REG_FULL
    IMG_MEAN = cfg.IMG_MEAN[None, :, None, None]
    IMG_STD = cfg.IMG_STD[None, :, None, None]
    THRESHOLD_CLS = cfg.THRESHOLD_CLS

    PIXEL_OFFSET_PER_FRAME = cfg.PIXEL_OFFSET_PER_FRAME

    global perform_inference, roi_defined, p1, p2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    model = SiameseTracker(SIZE_TEMPLATE, SIZE_SEARCH, SIZE_OUT, REG_FULL).to(device)
    model.load_state_dict(torch.load(MODEL_PATH_INFERENCE))
    model.eval()
    init_frame = False

    stride_search_out = SIZE_SEARCH / SIZE_OUT

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("Stream")
    cv2.setMouseCallback("Stream", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_img, w_img = frame.shape[:2]
        if init_frame == False:
            display_frame = np.zeros((max(h_img, SIZE_TEMPLATE+2*SIZE_SEARCH), w_img+SIZE_SEARCH, 3), dtype="uint8")
            init_frame = True
        display_frame[0:h_img, 0:w_img, :] = frame.copy()
        #display_frame = frame.copy()
        if perform_inference == False: # Get template image
            if drawing and p1:
                curr = current_mouse_pos
                cv2.rectangle(display_frame, p1, curr, (0, 255, 0), 2)
            elif roi_defined:
                cv2.rectangle(display_frame, p1, p2, (0, 255, 255), 2)
                if p1[0] == p2[0] or p1[1] == p2[1]:
                    print("You have to drag to draw a bounding box!")
                    roi_defined = False
                    p1 = None
                    p2 = None
                    continue
                if p2[0] > p1[0]:
                    x1 = p1[0]
                    x2 = p2[0]
                else:
                    x1 = p2[0]
                    x2 = p1[0]
                if p2[1] > p1[1]:
                    y1 = p1[1]
                    y2 = p2[1]
                else:
                    y1 = p2[1]
                    y2 = p1[1]
                bbox_template = [x1, y1, x2-x1, y2-y1]
                cx, cy, size = get_context_bbox(bbox_template, EXTRA_CONTENT)
                template_img, scale_template = crop_and_resize(frame, cx, cy, size, SIZE_TEMPLATE, 0, 0)
                display_frame[0:SIZE_TEMPLATE, w_img:w_img+SIZE_TEMPLATE] = template_img.copy()
                template_tensor = to_tensor(template_img, IMG_MEAN, IMG_STD).to(device, dtype=torch.float).unsqueeze(0)
                perform_inference = True

        else: # Get search image and start inference
            search_img, scale_search = crop_and_resize(frame, cx, cy, size, SIZE_SEARCH, 0, 0)
            display_frame[SIZE_TEMPLATE:SIZE_TEMPLATE+SIZE_SEARCH, w_img:w_img+SIZE_SEARCH] = search_img.copy()
            search_tensor = to_tensor(search_img, IMG_MEAN, IMG_STD).to(device, dtype=torch.float).unsqueeze(0)
            # Forward pass
            pred_heatmap, pred_bbox = model(template_tensor, search_tensor)
            bbox = pred_bbox[0].detach().cpu().numpy()
            heatmap = torch.sigmoid(pred_heatmap[0]).detach().cpu().numpy()

            # Step 1: Normalize to [0, 255]
            hm_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            # Step 2: Convert to uint8
            hm_uint8 = hm_normalized.astype(np.uint8)
            # Step 3: Resize using INTER_NEAREST (KNN-like)
            hm_resized = cv2.resize(hm_uint8, (SIZE_SEARCH, SIZE_SEARCH)) #, interpolation=cv2.INTER_NEAREST)
            # Step 4: Apply colormap (e.g., COLORMAP_HOT)
            heatmap_bgr = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)
            display_frame[SIZE_TEMPLATE+SIZE_SEARCH:SIZE_TEMPLATE+2*SIZE_SEARCH, w_img:w_img+SIZE_SEARCH] = heatmap_bgr.copy()

            if heatmap.max() > THRESHOLD_CLS:
                ci, cj = heatmap_center_of_mass(heatmap)
                cx_search = (cj + 0.5) * stride_search_out
                cy_search = (ci + 0.5) * stride_search_out
                w, h = bbox[ci, cj]
                w_search = w*SIZE_SEARCH
                h_search = h*SIZE_SEARCH
                x0, y0 = cx_search - w_search/2, cy_search - h_search/2
                x1, y1 = cx_search + w_search/2, cy_search + h_search/2
                cv2.rectangle(display_frame, [int(x0)+w_img, int(y0)+SIZE_TEMPLATE], [int(x1)+w_img, int(y1)+SIZE_TEMPLATE], color=(0, 255, 0), thickness=2)
            
                patch_search = [cx-size/2.0, cy-size/2.0, cx+size/2.0, cy+size/2.0]

                scale = size/SIZE_SEARCH
                scale = size/SIZE_SEARCH
                x0_img = patch_search[0] + x0*scale
                x1_img = patch_search[0] + x1*scale
                y0_img = patch_search[1] + y0*scale
                y1_img = patch_search[1] + y1*scale
                #cv2.rectangle(display_frame, [int(patch_search[0]), int(patch_search[1])], [int(patch_search[2]), int(patch_search[3])], color=(0, 255, 0), thickness=2)
                cv2.rectangle(display_frame, [int(x0_img), int(y0_img)], [int(x1_img), int(y1_img)], color=(0, 255, 0), thickness=2)

                desired_cx = patch_search[0] + cx_search*scale
                desired_cy = patch_search[1] + cy_search*scale

                if desired_cx > cx:
                    cx = min(cx+PIXEL_OFFSET_PER_FRAME, desired_cx)
                else:
                    cx = max(cx-PIXEL_OFFSET_PER_FRAME, desired_cx)
                if desired_cy > cy:
                    cy = min(cy+PIXEL_OFFSET_PER_FRAME, desired_cy)
                else:
                    cy = max(cy-PIXEL_OFFSET_PER_FRAME, desired_cy)

        cv2.imshow("Stream", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()