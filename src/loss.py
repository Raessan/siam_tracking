import torch
import torch.nn.functional as F

def compute_loss(pred_heat, pred_bbox, gt_heat, gt_bbox, alpha=0.25, gamma=2.0, weight=1.0):
    B, H, W = pred_heat.shape

    pred_sig = torch.sigmoid(pred_heat)
    pos_inds = gt_heat > 0
    neg_inds = gt_heat == 0

    pos_weight = alpha * (1 - pred_sig[pos_inds]) ** gamma
    neg_weight = (1 - alpha) * (pred_sig[neg_inds]) ** gamma

    loss_pos = F.binary_cross_entropy(pred_sig[pos_inds], gt_heat[pos_inds], reduction='none')
    loss_neg = F.binary_cross_entropy(pred_sig[neg_inds], gt_heat[neg_inds], reduction='none')

    cls_loss = (pos_weight * loss_pos).sum() + (neg_weight * loss_neg).sum()
    num_pos = pos_inds.sum().clamp(min=1).float()
    cls_loss = cls_loss / num_pos

    reg_mask = (gt_heat > 0).unsqueeze(-1).float()
    abs_diff = torch.abs(pred_bbox - gt_bbox) # To consider all coordinates
    reg_loss = (abs_diff * reg_mask).sum() / num_pos

    total_loss = cls_loss + weight * reg_loss
    return total_loss, cls_loss, reg_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup dummy data
    B, H, W = 1, 5, 5  # Small for clarity
    pred_heat = torch.full((B, H, W), -2.0).to(device)  # Low confidence
    pred_heat[0, 2, 2] = 2.0  # High confidence at center

    pred_wh = torch.zeros((B, H, W, 2)).to(device)
    pred_wh[0, 2, 2] = torch.tensor([0.5, 0.5])  # Predicted size at center

    gt_heat = torch.zeros((B, H, W)).to(device)
    gt_heat[0, 2, 2] = 1.0  # Ground-truth object center

    gt_wh = torch.zeros((B, H, W, 2)).to(device)
    gt_wh[0, 2, 2] = torch.tensor([0.5, 0.5])  # Ground-truth width/height

    # Compute losses
    total_loss, cls_loss, reg_loss = compute_loss(pred_heat, pred_wh, gt_heat, gt_wh)

    print("Total Loss:", total_loss.item())
    print("Classification Loss:", cls_loss.item())
    print("Regression Loss:", reg_loss.item())