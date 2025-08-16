import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ Backbone wrapper for DINO (or any ViT that can return intermediate layers) ------------
class DinoBackbone(nn.Module):
    """
    Wraps a DINO vision transformer to return spatial feature maps (B, C, H, W).
    Expects `dino_model.get_intermediate_layers(img, n=..., reshape=True, norm=True)` to be available.
    If your DINO API differs, replace `get_intermediate_layers` call accordingly.
    """
    def __init__(self, dino_model, n_layers = 12, proj_dim=256):
        """
        dino_model: pretrained DINO model instance
        layer_idx: which intermediate layer to extract (-1 for last)
        proj_dim: optional channel projection to reduce dimension for lightweight heads
        """
        super().__init__()
        self.dino = dino_model
        self.n_layers = n_layers

        # try to get feature dim by a dry run (optional)
        # but we'll leave it dynamic; we'll define a projection conv to proj_dim
        self.proj = None
        self.proj_dim = proj_dim
        # create a small 1x1 conv projection later lazily on first forward (to adapt to C)

    def forward(self, x):
        """
        x: (B, 3, H, W) in range expected by DINO (make sure to normalize as the DINO expects)
        returns: (B, proj_dim, Hf, Wf)
        """
        # Get intermediate layers. Many DINO versions provide this helper:
        feats = self.dino.get_intermediate_layers(x, n=range(self.n_layers),
                                                     reshape=True, norm=True)
        # get_intermediate_layers often returns a list; pick the element
        feat = feats[-1]  # (B, C, Hf, Wf)

        B, C, Hf, Wf = feat.shape
        if self.proj is None:
            # lazy init projection to reduce channel dimension for lightweight heads
            self.proj = nn.Conv2d(C, self.proj_dim, kernel_size=1, bias=True).to(feat.device)

        feat = self.proj(feat)  # (B, proj_dim, Hf, Wf)
        return feat
    
# ------------ Depthwise cross-correlation head (SiamFC style) ------------
class DepthwiseXCorrHead(nn.Module):
    """
    Depthwise cross-correlation between template and search features,
    then tiny conv heads for cls/reg.
    """
    def __init__(self, in_channels, mid_channels=256, reg_full=True):
        super().__init__()
        self.reg_full = reg_full

        # optional small conv to reduce channels before xcorr
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.mid_channels = mid_channels

        # classification & regression heads operate on xcorr output
        self.cls_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)
        )
        reg_dim = 4 if reg_full else 2
        self.reg_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, reg_dim, 1)
        )

    @staticmethod
    def _depthwise_xcorr_single(template, search):
        """
        template: (C, Ht, Wt)
        search:   (C, Hs, Ws)
        returns:  (C, Hout, Wout) depthwise xcorr (per-channel)
        Implementation uses grouped conv with batch collapsed trick.
        """
        C, Ht, Wt = template.shape
        _, Hs, Ws = search.shape
        # kernel shape expected: (C, 1, Ht, Wt)
        kernel = template.view(C, 1, Ht, Wt)
        # input needs to be (1, C, Hs, Ws)
        inp = search.unsqueeze(0)
        # perform grouped conv with groups=C -> out (1, C, Hout, Wout)
        out = F.conv2d(inp, kernel, groups=C)
        return out.squeeze(0)  # (C, Hout, Wout)

    def forward(self, feat_t, feat_s):
        """
        feat_t: (B, C, Ht, Wt)
        feat_s: (B, C, Hs, Ws)
        returns: cls_map (B, Hout, Wout), regs (B, Hout, Wout, reg_dim)
        """
        # reduce channels
        t = self.pre(feat_t)  # (B, M, Ht, Wt)
        s = self.pre(feat_s)  # (B, M, Hs, Ws)
        B, M, Ht, Wt = t.shape

        # compute depthwise xcorr per sample (could be accelerated vectorized)
        xcorr_outs = []
        for i in range(B):
            # (M, Hout, Wout)
            x = self._depthwise_xcorr_single(t[i], s[i])
            xcorr_outs.append(x)
        xcorr = torch.stack(xcorr_outs, dim=0)  # (B, M, Hout, Wout)

        cls = self.cls_head(xcorr)   # (B, 1, Hout, Wout)
        regs = self.reg_head(xcorr)  # (B, reg_dim, Hout, Wout)

        cls = cls.squeeze(1)  # (B, Hout, Wout)
        regs = regs.permute(0, 2, 3, 1)  # (B, Hout, Wout, reg_dim)
        return cls, regs
    
# ------------ Alternative simpler head: cosine correlation + conv ------------
class CosineCorrHead(nn.Module):
    """
    Compute cosine similarity between template pooled feature and search features
    (1x1 style) and then tiny conv heads. This is even cheaper but loses some
    spatial-template shape info.
    """
    def __init__(self, in_channels, mid_channels=256, reg_full=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.cls = nn.Conv2d(mid_channels, 1, 1)
        self.reg = nn.Conv2d(mid_channels, 4 if reg_full else 2, 1)

    def forward(self, feat_t, feat_s):
        # feat_t: (B, C, Ht, Wt) -> global pooled to (B, C, 1, 1)
        tpl = self.avgpool(feat_t)     # (B, C, 1, 1)
        tpl = self.reduce(tpl)         # (B, mid, 1, 1)
        s = self.reduce(feat_s)        # (B, mid, Hs, Ws)

        # cosine between template vector and every spatial location
        tpl_norm = F.normalize(tpl.view(tpl.size(0), tpl.size(1)), dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, mid,1,1)
        s_norm = F.normalize(s, dim=1)  # (B, mid, Hs, Ws)
        corr = (tpl_norm * s_norm).sum(dim=1)  # (B, Hs, Ws) -- dot over channel dim

        cls = self.cls(s)  # (B,1,Hs,Ws)  (you may prefer to feed corr into better head)
        reg = self.reg(s)
        cls = cls.squeeze(1)
        reg = reg.permute(0, 2, 3, 1)
        return corr, reg
    
# ------------ Putting it together: SiameseTracker with DINO backbone and chosen head ------------
class SiameseTrackerDino(nn.Module):
    def __init__(self, dino_model, head_type='depthwise', proj_dim=256, reg_full=True):
        """
        dino_model: pretrained DINO (ViT) object
        head_type: 'depthwise' (recommended) or 'cosine'
        proj_dim: number of channels after projecting DINO features
        """
        super().__init__()
        self.backbone = DinoBackbone(dino_model, proj_dim=proj_dim)
        C = proj_dim
        if head_type == 'depthwise':
            self.head = DepthwiseXCorrHead(in_channels=C, mid_channels=128, reg_full=reg_full)
        elif head_type == 'cosine':
            self.head = CosineCorrHead(in_channels=C, mid_channels=128, reg_full=reg_full)
        else:
            raise ValueError("head_type must be 'depthwise' or 'cosine'")
        
    def forward(self, template, search):
        # template, search shapes: (B, 3, H_in, W_in)
        feat_t = self.backbone(template)  # (B, C, Ht, Wt)
        feat_s = self.backbone(search)    # (B, C, Hs, Ws)
        cls_map, regs = self.head(feat_t, feat_s)
        return cls_map, regs
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template_size= 128
    search_size = 256
    out_size = 128
    template = torch.randn(32,3,template_size,template_size).to(device)
    search = torch.randn(32,3,search_size,search_size).to(device)
    dinov3_dir = "/home/rafa/deep_learning/projects/siam_tracking/dinov3"
    dino_model = torch.hub.load(
        repo_or_dir=dinov3_dir,
        model="dinov3_vits16plus",
        source="local"
    )
    model = SiameseTrackerDino(dino_model).to(device)
    # print(model)
    # feat = model(template)
    # print(feat.shape)
    # feat2 = model(search)
    # print(feat2.shape)
    cls, wh = model(template, search)
    print("Cls shape: ", cls.shape)
    print("wh shape: ", wh.shape)
    n_params = sum([p.numel() for p in model.parameters()])
    print("Total number of parameters: ", n_params)
    n_params_backbone = sum([p.numel() for p in model.backbone.parameters()])
    print("Number parameters backbone: ", n_params_backbone)
    n_params_cross_attn = sum([p.numel() for p in model.cross_attn.parameters()])
    print("Number parameters cross attn: ", n_params_cross_attn)
    n_params_cls = sum([p.numel() for p in model.cls_head.parameters()])
    print("Number parameters classification: ", n_params_cls)
    n_params_reg = sum([p.numel() for p in model.reg_head.parameters()])
    print("Number parameters regression: ", n_params_reg)
