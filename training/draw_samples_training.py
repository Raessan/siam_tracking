import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches

def draw_samples_training(template, search, heatmap, reg_wh, gt_heatmap, gt_reg_wh,
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), thresh_cls = 0.5,
                 n_samples_plot=4, video_template_names="", video_search_names=""):
    """
    template: Tensor[B,3,127,127]
    search:   Tensor[B,3,255,255]
    heatmap:  Tensor[B,25,25]
    reg_wh:   Tensor[B,25,25,2]  (w,h in pixels)
    mean,std: tuples for denormalization
    """
    B = template.shape[0]
    idxs = random.sample(range(B), min(B, n_samples_plot))
    
    # prepare denorm
    mean = torch.tensor(mean, device=template.device).view(1,3,1,1)
    std  = torch.tensor(std,  device=template.device).view(1,3,1,1)
    
    stride = search.shape[-1] / heatmap.shape[-1]  # e.g. 255/25 = 10.2
    
    n = len(idxs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    for row, i in enumerate(idxs):
        # 1) denormalize template
        tpl = template[i:i+1] * std + mean
        tpl = tpl.clamp(0,1).cpu().numpy()[0].transpose(1,2,0)
        
        # 2) denormalize search
        sr = search[i:i+1] * std + mean
        sr = sr.clamp(0,1).cpu().numpy()[0].transpose(1,2,0)
        
        # 3) heatmap & reg_wh
        hm = heatmap[i].detach().cpu().numpy()        # (25,25)
        wh = reg_wh[i].detach().cpu().numpy()         # (25,25,2)
        gt_hm = gt_heatmap[i].detach().cpu().numpy()
        gt_wh = gt_reg_wh[i].detach().cpu().numpy()
        
        # find best cell of the predicted
        if hm.max() > thresh_cls:
            idx = np.unravel_index(hm.argmax(), hm.shape)
            ci, cj = idx  # row, col in heatmap
            cx = (cj + 0.5) * stride
            cy = (ci + 0.5) * stride
            w, h = wh[ci, cj]
            w *= search.shape[2]
            h *= search.shape[3]
            x0, y0 = cx - w/2, cy - h/2
        else:
            x0 = 0
            y0 = 0
            w = 0
            h = 0

        # find best cell of the gt
        if gt_hm.max() > thresh_cls:
            gt_idx = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
            gt_ci, gt_cj = gt_idx  # row, col in heatmap
            gt_cx = (gt_cj + 0.5) * stride
            gt_cy = (gt_ci + 0.5) * stride
            gt_w, gt_h = gt_wh[gt_ci, gt_cj]
            gt_w *= search.shape[2]
            gt_h *= search.shape[3]
            gt_x0, gt_y0 = gt_cx - gt_w/2, gt_cy - gt_h/2
        else:
            gt_x0 = 0
            gt_y0 = 0
            gt_w = 0
            gt_h = 0
        
        # Plot template
        ax = axes[row,0] if n>1 else axes[0]
        ax.imshow(tpl)
        ax.set_title(f"Video {video_template_names[i] if video_template_names!='' else ''}")
        ax.axis('off')
        
        # Plot search + box
        ax = axes[row,1] if n>1 else axes[1]
        ax.imshow(sr)
        rect = patches.Rectangle((x0,y0), w, h,
                                  linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        gt_rect = patches.Rectangle((gt_x0,gt_y0), gt_w, gt_h,
                                  linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(gt_rect)
        ax.set_title(f"Video {video_search_names[i] if video_search_names!='' else ''}")
        ax.axis('off')
        
        # Plot heatmap
        ax = axes[row,2] if n>1 else axes[2]
        im = ax.imshow(hm, cmap='hot', origin='upper',
                       interpolation='nearest')
        ax.set_title("Heatmap")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()