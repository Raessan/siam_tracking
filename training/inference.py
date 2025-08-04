import torch
import cv2
import numpy as np
from model import SiameseTracker
from utils import get_context_bbox, crop_and_resize, to_tensor, heatmap_center_of_mass

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

    MODEL_PATH = 'results/2025-08-02_20-03-42/model_0.pth'  # Replace with your actual model path
    EXTRA_CONTENT = 0.25
    SEARCH_SIZE = 255
    TEMPLATE_SIZE = 127
    OUT_SIZE = 25
    REG_FULL = False
    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None,:,None,None]
    IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None,:,None,None]
    THRESHOLD_CLS = 0.5

    global perform_inference, roi_defined, p1, p2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    model = SiameseTracker(TEMPLATE_SIZE, SEARCH_SIZE, OUT_SIZE, REG_FULL).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    init_frame = False

    stride_search_out = SEARCH_SIZE / OUT_SIZE

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
            display_frame = np.zeros((max(h_img, TEMPLATE_SIZE+2*SEARCH_SIZE), w_img+SEARCH_SIZE, 3), dtype="uint8")
            init_frame = True
        display_frame[0:h_img, 0:w_img, :] = frame.copy()
        #display_frame = frame.copy()
        if perform_inference == False: # Get template image
            if drawing and p1:
                curr = current_mouse_pos
                cv2.rectangle(display_frame, p1, curr, (0, 255, 0), 2)
            elif roi_defined:
                cv2.rectangle(display_frame, p1, p2, (0, 255, 255), 2)
                print("p1, p2: ", p1, p2)
                if p2[0] > p1[0] or p2[1] > p1[1]: # Exchange to ensure that p1 is top left corner
                    bbox_template = [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]]
                elif p2[0] < p1[0] or p2[1] < p1[1]: # Exchange to ensure that p1 is top left corner
                    bbox_template = [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]]
                else:
                    print("You have to drag to draw a bounding box!")
                    roi_defined = False
                    p1 = None
                    p2 = None
                    continue
                cx, cy, size = get_context_bbox(bbox_template, EXTRA_CONTENT)
                template_img, scale_template, patch_template = crop_and_resize(frame, cx, cy, size, TEMPLATE_SIZE, 0, 0)
                display_frame[0:TEMPLATE_SIZE, w_img:w_img+TEMPLATE_SIZE] = template_img.copy()
                template_tensor = to_tensor(template_img, IMG_MEAN, IMG_STD).to(device, dtype=torch.float).unsqueeze(0)
                perform_inference = True

        else: # Get search image and start inference
            search_img, scale_search, patch_search = crop_and_resize(frame, cx, cy, size, SEARCH_SIZE, 0, 0)
            print("patch_search: ", patch_search)
            display_frame[TEMPLATE_SIZE:TEMPLATE_SIZE+SEARCH_SIZE, w_img:w_img+SEARCH_SIZE] = search_img.copy()
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
            hm_resized = cv2.resize(hm_uint8, (SEARCH_SIZE, SEARCH_SIZE)) #, interpolation=cv2.INTER_NEAREST)
            # Step 4: Apply colormap (e.g., COLORMAP_HOT)
            heatmap_bgr = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)
            display_frame[TEMPLATE_SIZE+SEARCH_SIZE:TEMPLATE_SIZE+2*SEARCH_SIZE, w_img:w_img+SEARCH_SIZE] = heatmap_bgr.copy()

            if heatmap.max() > THRESHOLD_CLS:
                ci, cj = heatmap_center_of_mass(heatmap)
                cx_search = (cj + 0.5) * stride_search_out
                cy_search = (ci + 0.5) * stride_search_out
                w, h = bbox[ci, cj]
                w_search = w*SEARCH_SIZE
                h_search = h*SEARCH_SIZE
                x0, y0 = cx_search - w_search/2, cy_search - h_search/2
                x1, y1 = cx_search + w_search/2, cy_search + h_search/2
                cv2.rectangle(display_frame, [int(x0)+w_img, int(y0)+TEMPLATE_SIZE], [int(x1)+w_img, int(y1)+TEMPLATE_SIZE], color=(0, 255, 0), thickness=2)
            
                w_patch = patch_search[2] - patch_search[0]
                h_patch = patch_search[3] - patch_search[1]
                
                scale_w = w_patch/SEARCH_SIZE
                scale_h = h_patch/SEARCH_SIZE
                w_bbox_img = w_search*scale_w
                h_bbox_img = h_search*scale_h
                print("w, h: ", w_img, h_img)
                print("patch_search: ", patch_search)
                cx_img = patch_search[2] + cx_search*scale_w
                cy_img = patch_search[3] + cy_search*scale_h
                cv2.rectangle(display_frame, [int((cx_img-w_bbox_img)/2), int((cy_img-h_bbox_img)/2)], [int((cx_img+w_bbox_img)/2), int((cy_img+h_bbox_img)/2)], color=(0, 255, 0), thickness=2)

                
            # stride para ponerlo en la foto grande, actualizar cx, cy



        cv2.imshow("Stream", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()