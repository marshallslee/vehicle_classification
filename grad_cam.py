import os, argparse, torch, cv2
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import build_model

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, required=True, choices=["baseline","resnet50","efficientnet_b0"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_samples", type=int, default=6)
    return p.parse_args()

def choose_target_layer(model, arch):
    if arch == "resnet50":
        return model.layer4[-1].conv3
    if arch == "efficientnet_b0":
        return model.features[-1][0]
    return model.features[-1]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    def save_activation(self, module, inp, out):
        self.activations = out.detach()
    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def __call__(self, scores, class_idx):
        self.model.zero_grad()
        loss = scores[0, class_idx]; loss.backward(retain_graph=True)
        grads = self.gradients; acts = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam

def overlay_cam(img_rgb, cam):
    h, w, _ = img_rgb.shape
    cam = cv2.resize(cam, (w,h))
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.4*heatmap + 0.6*img_rgb).astype(np.uint8)
    return overlay

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = args.arch; class_to_idx = ckpt["class_to_idx"]; idx_to_class = {v:k for k,v in class_to_idx.items()}

    model = build_model(arch, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(os.path.join(args.data_dir,"test"), transform=tf)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    layer = choose_target_layer(model, arch)
    cam = GradCAM(model, layer)

    out_dir = os.path.dirname(args.checkpoint); os.makedirs(out_dir, exist_ok=True)
    n = args.num_samples
    done = 0
    for (xb, yb), (path, _) in zip(loader, ds.samples):
        xb = xb.to(device)
        logits = model(xb); probs = torch.softmax(logits, dim=1)
        cls = torch.argmax(probs, dim=1).item()
        _ = model(xb)  # forward again for hooks
        cam_map = cam(logits, cls)

        img_bgr = cv2.imread(path); img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (args.img_size, args.img_size))
        overlay = overlay_cam(img_rgb, cam_map)

        out_path = os.path.join(out_dir, f"gradcam_{done+1}_{idx_to_class.get(cls,'cls')}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)
        done += 1
        if done >= n: break

if __name__ == "__main__":
    main()
