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
    def __init__(self, dino_model, n_layers = 12):
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
        #self.proj_dim = proj_dim
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
        # if self.proj is None:
        #     # lazy init projection to reduce channel dimension for lightweight heads
        #     self.proj = nn.Conv2d(C, self.proj_dim, kernel_size=1, bias=True).to(feat.device)

        # feat = self.proj(feat)  # (B, proj_dim, Hf, Wf)
        return feat
    
class CrossAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out   = nn.Conv2d(dim, dim, 1)

    def forward(self, feat_tpl, feat_srch):
        # Query = search, Key/Value = template -> update search
        B, C, Hs, Ws = feat_srch.shape
        _, _, Ht, Wt = feat_tpl.shape

        q = self.q_proj(feat_srch).flatten(2)   # (B, C, N_s)
        k = self.k_proj(feat_tpl).flatten(2)    # (B, C, N_t)
        v = self.v_proj(feat_tpl).flatten(2)    # (B, C, N_t)

        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)

        attn = (q.transpose(-2, -1) @ k) * self.scale   # (B, h, N_s, N_t)
        attn = attn.softmax(dim=-1)

        out = attn @ v.transpose(-2, -1)  # (B, h, N_s, d)
        out = out.transpose(-2, -1).contiguous().view(B, C, -1)
        out = out.view(B, C, Hs, Ws)

        return feat_srch + self.out(out)
    
# ------------ Putting it together: SiameseTracker with DINO backbone and chosen head ------------
class SiameseTrackerDino(nn.Module):
    def __init__(self, dino_model, n_layers_dino, embed_dim, out_size, proj_dim=256, reg_full=True):
        """
        dino_model: pretrained DINO (ViT) object
        head_type: 'depthwise' (recommended) or 'cosine'
        proj_dim: number of channels after projecting DINO features
        """
        super().__init__()
        self.backbone = DinoBackbone(dino_model, n_layers = n_layers_dino)
        C = proj_dim
        self.reg_full = reg_full
        if reg_full:
            reg_dim = 4
        else:
            reg_dim = 2

        # Projection layer
        self.proj = nn.Conv2d(embed_dim, proj_dim, kernel_size=1, bias=True)

        # cross‐attention: template→search
        self.cross_attn = CrossAttentionModule(dim=C, num_heads=8)

        # feature fusion
        self.fuse = nn.Conv2d(C, C, 1)

        # heads: classification and wh regression
        self.cls_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(C, 1, 1)  # single‐channel heatmap
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(C, reg_dim, 1),  # four‐channel (dx,dy,w,h) or two-channel (w,h)
        )

        # output feature map size (H_out, W_out)
        self.out_size = out_size
        
        
    def forward(self, template, search):

        # 1) extract features
        f_t = self.backbone(template)  # (B, C, Ht, Wt)
        f_s = self.backbone(search)    # (B, C, Hs, Ws)

        # 2) Project to proj_dim
        f_t = self.proj(f_t)
        f_s = self.proj(f_s)

        # 3) attend search with template
        f_s = self.cross_attn(f_t, f_s)

        # 4) fuse / reduce
        f_s = self.fuse(f_s)  # (B, C, Hout, Wout)

        # 5) predict
        cls = self.cls_head(f_s)        # (B, 1, Hout, Wout)
        regs  = self.reg_head(f_s)         # (B, 2 or 4, Hout, Wout)

        # resize to exact out_size if needed
        cls = F.interpolate(cls, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

        if self.reg_full:
            # --- split off offsets vs sizes ---
            # regs channels: [0]=dx, [1]=dy, [2]=w, [3]=h
            dxdy, wh = regs.split([2, 2], dim=1)
            wh = F.softplus(wh)             # enforce w,h ≥ 0

            # recombine
            regs = torch.cat([dxdy, wh], dim=1)  # (B,4,Hout,Wout)

        else:
            regs = F.softplus(regs)

        # resize to exact out_size if needed    
        regs  = F.interpolate(regs,  size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

        # squeeze channel dims
        cls = cls.squeeze(1)            # (B, Hout, Wout)
        regs  = regs.permute(0,2,3,1)       # (B, Hout, Wout, 4)

        return cls, regs
    

# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5

#         # linear projections for template (query) and search (key/value)
#         self.q_proj = nn.Conv2d(dim, dim, 1)
#         self.k_proj = nn.Conv2d(dim, dim, 1)
#         self.v_proj = nn.Conv2d(dim, dim, 1)
#         self.out   = nn.Conv2d(dim, dim, 1)

#     def forward(self, feat_tpl, feat_srch):
#         B, C, H, W = feat_srch.shape

#         # flatten spatial dims
#         q = self.q_proj(feat_tpl).flatten(2)   # (B, C, N1)
#         k = self.k_proj(feat_srch).flatten(2)  # (B, C, N2)
#         v = self.v_proj(feat_srch).flatten(2)  # (B, C, N2)

#         # reshape for multihead
#         q = q.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N1)
#         k = k.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N2)
#         v = v.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N2)

#         attn = (q.transpose(-2,-1) @ k) * self.scale         # (B, h, N1, N2)
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v.transpose(-2,-1))                   # (B, h, N1, d)
#         out = out.transpose(-2,-1).contiguous().view(B, C, -1)  # (B, C, N1)
#         out = out.view(B, C, feat_tpl.shape[-2], feat_tpl.shape[-1])
#         return feat_tpl + self.out(out)  # residual

# # ------------ Depthwise cross-correlation head (SiamFC style) ------------
# class DepthwiseXCorrHead(nn.Module):
#     def __init__(self, in_channels, out_size, mid_channels=256, reg_full=True):
#         super().__init__()
#         self.reg_full = reg_full
#         self.out_size = out_size

#         self.pre = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.mid_channels = mid_channels

#         self.cls_head = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, 1, 1)   # logits for heatmap
#         )
#         reg_dim = 4 if reg_full else 2
#         self.reg_head = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, reg_dim, 1)
#         )

#     @staticmethod
#     def _depthwise_xcorr_single(template, search):
#         C, Ht, Wt = template.shape
#         _, Hs, Ws = search.shape
#         kernel = template.view(C, 1, Ht, Wt)
#         inp = search.unsqueeze(0)  # (1, C, Hs, Ws)
#         out = F.conv2d(inp, kernel, groups=C)  # (1, C, Hout, Wout)
#         return out.squeeze(0)  # (C, Hout, Wout)

#     def forward(self, feat_t, feat_s):
#         # feat_t: (B, C, Ht, Wt), feat_s: (B, C, Hs, Ws)
#         t = self.pre(feat_t)  # (B, M, Ht, Wt)
#         s = self.pre(feat_s)  # (B, M, Hs, Ws)
#         B, M, Ht, Wt = t.shape

#         # compute depthwise xcorr: per-sample loop (vectorize if needed)
#         xcorr_outs = []
#         for i in range(B):
#             x = self._depthwise_xcorr_single(t[i], s[i])  # (M, Hout, Wout)
#             xcorr_outs.append(x)
#         xcorr = torch.stack(xcorr_outs, dim=0)  # (B, M, Hout, Wout)

#         cls = self.cls_head(xcorr)   # (B, 1, Hout, Wout)
#         regs = self.reg_head(xcorr)  # (B, reg_dim, Hout, Wout)

#         # upsample to requested out_size (25x25)
#         cls = F.interpolate(cls, size=(self.out_size, self.out_size),
#                             mode="bilinear", align_corners=False)  # (B,1,Out,Out)
#         regs = F.interpolate(regs, size=(self.out_size, self.out_size),
#                              mode="bilinear", align_corners=False)  # (B,reg_dim,Out,Out)

#         cls = cls.squeeze(1)  # (B, Out, Out)
#         regs = regs.permute(0, 2, 3, 1)  # (B, Out, Out, reg_dim)

#         # For positivity on widths/heights (if reg_full), apply softplus to the corresponding channels:
#         if self.reg_full:
#             # regs[..., 2:4] correspond to w,h - ensure they are positive
#             regs_wh = F.softplus(regs[..., 2:4])
#             regs = torch.cat([regs[..., :2], regs_wh], dim=-1)
#         else:
#             regs = F.softplus(regs)

#         return cls, regs
    
# # ------------ Alternative simpler head: cosine correlation + conv ------------
# class CosineCorrHead(nn.Module):
#     def __init__(self, in_channels, out_size, mid_channels=256, reg_full=True):
#         super().__init__()
#         self.out_size = out_size
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
#         self.cls_conv = nn.Conv2d(mid_channels, 1, 1)
#         self.reg_conv = nn.Conv2d(mid_channels, 4 if reg_full else 2, 1)
#         self.reg_full = reg_full

#     def forward(self, feat_t, feat_s):
#         tpl = self.avgpool(feat_t)     # (B, C, 1, 1)
#         tpl = self.reduce(tpl)         # (B, mid, 1, 1)
#         s = self.reduce(feat_s)        # (B, mid, Hs, Ws)

#         tpl_norm = F.normalize(tpl.view(tpl.size(0), tpl.size(1)), dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, mid,1,1)
#         s_norm = F.normalize(s, dim=1)  # (B, mid, Hs, Ws)
#         corr = (tpl_norm * s_norm).sum(dim=1, keepdim=True)  # (B, 1, Hs, Ws) -- dot over channel dim

#         # Option A: use corr directly as cls logits, but we also provide small conv head on s
#         cls_logits = self.cls_conv(s)  # (B,1,Hs,Ws)  (alternative: use corr)
#         reg = self.reg_conv(s)         # (B,reg_dim,Hs,Ws)

#         # Upsample to out_size
#         cls_logits = F.interpolate(cls_logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
#         reg = F.interpolate(reg, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        
#         cls = cls_logits.squeeze(1)           # (B, Out, Out)
#         reg = reg.permute(0, 2, 3, 1)         # (B, Out, Out, reg_dim)

#         if self.reg_full:
#             reg_wh = F.softplus(reg[..., 2:4])
#             reg = torch.cat([reg[..., :2], reg_wh], dim=-1)
#         else:
#             reg = F.softplus(reg)

#         # optionally return corr (upsampled) too if you want a similarity map:
#         # corr_up = F.interpolate(corr, size=(self.out_size,self.out_size), mode='bilinear', align_corners=False).squeeze(1)

#         return cls, reg
    
# ------------ Putting it together: SiameseTracker with DINO backbone and chosen head ------------
# class SiameseTrackerDino(nn.Module):
#     def __init__(self, dino_model, n_layers_dino, out_size, head_type='depthwise', proj_dim=256, reg_full=True):
#         """
#         dino_model: pretrained DINO (ViT) object
#         head_type: 'depthwise' (recommended) or 'cosine'
#         proj_dim: number of channels after projecting DINO features
#         """
#         super().__init__()
#         self.backbone = DinoBackbone(dino_model, n_layers = n_layers_dino, proj_dim=proj_dim)
#         C = proj_dim
#         if head_type == 'depthwise':
#             self.head = DepthwiseXCorrHead(in_channels=C, out_size = out_size, mid_channels=128, reg_full=reg_full)
#         elif head_type == 'cosine':
#             self.head = CosineCorrHead(in_channels=C, out_size = out_size, mid_channels=128, reg_full=reg_full)
#         else:
#             raise ValueError("head_type must be 'depthwise' or 'cosine'")
        
#     def forward(self, template, search):
#         # template, search shapes: (B, 3, H_in, W_in)
#         feat_t = self.backbone(template)  # (B, C, Ht, Wt)
#         feat_s = self.backbone(search)    # (B, C, Hs, Ws)
#         print(feat_t.shape)
#         print(feat_s.shape)
#         cls_map, regs = self.head(feat_t, feat_s)
#         return cls_map, regs
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template_size= 128
    search_size = 256
    out_size = 25
    template = torch.randn(32,3,template_size,template_size).to(device)
    search = torch.randn(32,3,search_size,search_size).to(device)
    dinov3_dir = "/home/rafa/deep_learning/projects/siam_tracking/dinov3"
    dino_model = torch.hub.load(
        repo_or_dir=dinov3_dir,
        model="dinov3_vits16plus",
        source="local"
    )
    n_layers_dino = 12
    embed_dim = 384
    model = SiameseTrackerDino(dino_model, n_layers_dino, embed_dim, out_size, proj_dim = 512).to(device)
    cls, wh = model(template, search)
    print("Cls shape: ", cls.shape)
    print("wh shape: ", wh.shape)
    n_params = sum([p.numel() for p in model.parameters()])
    print("Total number of parameters: ", n_params)
    n_params_backbone = sum([p.numel() for p in model.backbone.parameters()])
    print("Number parameters backbone: ", n_params_backbone)