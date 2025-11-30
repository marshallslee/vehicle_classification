import cv2
import torch
import time
import argparse
import numpy as np
from torchvision import transforms
from models import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, required=True,
                   choices=["baseline", "resnet50", "efficientnet_b0"])
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--img_size", type=int, default=224)

    # 너무 작은 contour는 무시
    p.add_argument("--min_area", type=int, default=15000,
                   help="이 면적보다 작은 박스는 무시 (노이즈 제거용)")

    # 분류 확률이 이 값보다 낮으면 박스를 안 그림
    p.add_argument("--conf_thresh", type=float, default=0.8,
                   help="이 확률 미만은 무시")

    # 프레임에서 세로 방향 관심영역 (0~1 비율)
    p.add_argument("--roi_ymin", type=float, default=0.5,
                   help="세로 방향에서 이 비율 위는 무시 (예: 0.5면 아래 50%만 사용)")
    p.add_argument("--roi_ymax", type=float, default=1.0,
                   help="세로 방향에서 이 비율까지 사용 (기본 1.0 = 맨 아래)")
    return p.parse_args()


def build_classifier(ckpt_path, arch, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(arch, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, idx_to_class


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1) 분류기 로드
    model, idx_to_class = build_classifier(args.checkpoint, args.arch, device)

    # 2) 전처리(transform) 정의 (Resize는 cv2에서 처리)
    tf = transforms.Compose([
        transforms.ToTensor(),  # (H,W,C, uint8) [0,255] -> (C,H,W) [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 3) 비디오 소스 열기
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    # 4) 배경 제거기
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=True
    )

    fps_time = time.time()
    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            if first_frame:
                print("[ERROR] 첫 프레임을 읽지 못했습니다. 비디오 경로/코덱을 확인하세요.")
            break
        first_frame = False

        h, w = frame.shape[:2]

        # ---- (1) 배경 제거해서 전경 마스크 얻기 ----
        fg_mask = backSub.apply(frame)

        # ---- (1-0) 그림자 제거: 200 이상만 전경으로 취급 (0=배경, 127=그림자, 255=전경)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # ---- (1-1) 관심영역(ROI) 적용: 도로가 있는 아래쪽만 보기 ----
        roi_mask = np.zeros_like(fg_mask)
        y_min = int(h * args.roi_ymin)
        y_max = int(h * args.roi_ymax)
        roi_mask[y_min:y_max, :] = 255
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # 그림자/노이즈 줄이기용 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ---- (2) 컨투어(윤곽선) 찾기 ----
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ---- (3) 각 컨투어를 박스로 만들고, 분류기 돌리기 ----
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            area = w_box * h_box
            if area < args.min_area:
                # 너무 작은 건 노이즈로 취급
                continue

            # 프레임 전체 대비 너무 큰 덩어리(거의 전체 프레임)도 무시
            if area > 0.8 * w * h:
                continue

            # 차량 가로세로 비율: 너무 세로로 긴 건 사람/폴 가능성이 높음
            aspect_ratio = w_box / float(h_box + 1e-6)
            if aspect_ratio < 1.1:   # 차는 보통 가로가 세로보다 길다
                continue
            if aspect_ratio > 5.0:   # 너무 긴 것도 이상함
                continue

            # 너무 좁은 폭/너무 넓은 폭 제한 (프레임 기준)
            if w_box < 0.05 * w or w_box > 0.8 * w:
                continue
            if h_box < 0.05 * h or h_box > 0.7 * h:
                continue

            # 프레임 경계를 벗어나지 않도록 클램프
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w_box)
            y2 = min(frame.shape[0], y + h_box)

            # ---- (3-1) 박스 안에 실제 전경 픽셀이 얼마나 있는지 확인 ----
            mask_roi = fg_mask[y1:y2, x1:x2]
            fg_pixels = cv2.countNonZero(mask_roi)
            fill_ratio = fg_pixels / float(area + 1e-6)
            # 거의 비어 있으면 노이즈로 봄
            if fill_ratio < 0.3:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # BGR -> RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_rgb = cv2.resize(roi_rgb, (args.img_size, args.img_size))

            # numpy -> tensor
            x_t = tf(roi_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x_t)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                cls_idx = int(probs.argmax())
                label_name = idx_to_class[cls_idx]
                conf = float(probs[cls_idx])

            # 확률이 너무 낮으면 그리지 않기
            if conf < args.conf_thresh:
                continue

            label = f"{label_name} {conf*100:.1f}%"

            # ---- (4) 원본 프레임에 박스 + 라벨 그리기 ----
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # ---- (5) FPS 표시 ----
        fps = 1.0 / max(1e-6, (time.time() - fps_time))
        fps_time = time.time()
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # 디버깅용으로 마스크도 같이 보고 싶으면 아래 한 줄 추가
        # cv2.imshow("fg_mask", fg_mask)

        cv2.imshow("Vehicle Detection + Classification (demo)", frame)
        # ESC 키: 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
