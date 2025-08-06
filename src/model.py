import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class SiameseBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet‑50 up through layer4
        backbone = resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4→4
        self.layer2 = backbone.layer2  # 4→8
        self.layer3 = backbone.layer3  # 8→16
        #self.layer4 = backbone.layer4  # 16→32

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # optionally fuse layer4 if you want higher‐level context
        # x4 = self.layer4(x); x = x + F.interpolate(x4, size=x.shape[-2:])
        return x  # e.g. (B, 1024, H/16, W/16)

class CrossAttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # linear projections for template (query) and search (key/value)
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out   = nn.Conv2d(dim, dim, 1)

    def forward(self, feat_tpl, feat_srch):
        B, C, H, W = feat_srch.shape

        # flatten spatial dims
        q = self.q_proj(feat_tpl).flatten(2)   # (B, C, N1)
        k = self.k_proj(feat_srch).flatten(2)  # (B, C, N2)
        v = self.v_proj(feat_srch).flatten(2)  # (B, C, N2)

        # reshape for multihead
        q = q.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N1)
        k = k.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N2)
        v = v.view(B, self.num_heads, C//self.num_heads, -1)  # (B, h, d, N2)

        attn = (q.transpose(-2,-1) @ k) * self.scale         # (B, h, N1, N2)
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2,-1))                   # (B, h, N1, d)
        out = out.transpose(-2,-1).contiguous().view(B, C, -1)  # (B, C, N1)
        out = out.view(B, C, feat_tpl.shape[-2], feat_tpl.shape[-1])
        return feat_tpl + self.out(out)  # residual

class SiameseTracker(nn.Module):
    def __init__(self, template_size, search_size, out_size, reg_full = True):
        super().__init__()
        # backbones
        self.backbone = SiameseBackbone()
        C = 1024  # resnet50 layer3 channels
        self.reg_full = reg_full
        if reg_full:
            reg_dim = 4
        else:
            reg_dim = 2

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
            #nn.Softplus()  # To ensure it is >0  
        )

        # output feature map size (H_out, W_out)
        self.out_size = out_size
        self.search_size = search_size
        self.template_size = template_size

    def forward(self, template, search):
        # 1) extract features
        f_t = self.backbone(template)  # (B, C, Ht, Wt)
        f_s = self.backbone(search)    # (B, C, Hs, Ws)

        # 2) attend search with template
        f_s = self.cross_attn(f_t, f_s)

        # 3) fuse / reduce
        f_s = self.fuse(f_s)  # (B, C, Hout, Wout)

        # 4) predict
        cls = self.cls_head(f_s)        # (B, 1, Hout, Wout)
        regs  = self.reg_head(f_s)         # (B, 4, Hout, Wout)

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
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template_size= 127
    search_size = 255
    out_size = 127
    template = torch.randn(32,3,template_size,template_size).to(device)
    search = torch.randn(32,3,out_size,out_size).to(device)
    model = SiameseTracker(template_size, search_size, out_size, True).to(device)
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
