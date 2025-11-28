import cv2, torch, time, argparse
from torchvision import transforms
from models import build_model

IMG_SIZE = 224

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, required=True, choices=["baseline","resnet50","efficientnet_b0"])
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--class_names", nargs="+", default=["car","truck","bus"])
    p.add_argument("--img_size", type=int, default=224)
    return p.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = args.arch
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(arch, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))

        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            cls = probs.argmax()
            label = f"{idx_to_class[cls]} {probs[cls]*100:.1f}%"

        fps = 1.0 / max(1e-6, (time.time() - fps_time))
        fps_time = time.time()

        cv2.putText(frame, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Vehicle Classification (demo)", frame)

        # ESC 키 누르면 아웃
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()