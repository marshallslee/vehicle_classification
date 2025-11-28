# 실행 방법

## 1. 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 데이터셋 구축 (Stanford + CompCars + extra truck/bus)

### (1) Stanford Cars를 data_3cls에 병합
```bash
python prepare_dataset.py \
  --stanford_root "./Stanford Cars Dataset" \
  --out "./data_3cls" \
  --crop_bbox 
```

* `crop_bbox`: 바운딩 박스를 이용해 차량 부분만 잘라내도록 함.

### (2) CompCars를 data_3cls에 병합
```bash
python prepare_dataset.py \
  --compcars_root "./CompCars Dataset" \
  --out "./data_3cls" \
  --crop_bbox 
```

### (3) 추가 트럭 및 버스 이미지 반영
```bash
python prepare_dataset.py \
  --bus_dir   "./extra datasets/bus" \
  --truck_dir "./extra datasets/truck" \
  --out "./data_3cls" 
```

```bash
python prepare_dataset.py \
  --truck_dir "./extra datasets 2/truck" \
  --out "./data_3cls" 
```

## 4. Baseline CNN 학습
```bash
python train.py \
  --data_dir ./data_3cls \
  --arch baseline \
  --epochs 30 \
  --weighted_loss \
  --use_weighted_sampler \
  --mixup \
  --mixup_alpha 0.2
```

* `batch_size`
  - 한 번의 학습 단계에서 모델이 처리하는 이미지 묶음의 크기. 
  - 즉 몇 장씩 묶어서 모델이 학습하느냐를 정의.
  - 배치 값이 큰 경우 특징
    - GPU 사용량이 많음
    - 안정적 학습
    - gradient가 평균화 되어 성능이 부드럽게 수렴
  - 배치 값이 작은 경우 특징
    - 메모리 사용량이 적음
    - gradient noise 때문에 불안정
* `weighted_loss`: 클래스별 데이터 수 차이 (불균형)를 보정하기 위해 클래스마다 다른 가중치를 부여하는 방식.
* `use_weighted_sampler`: 학습 데이터 로딩시 희소 클래스 (truck, bus)를 더 자주 샘플링 하도록 하는 sampler 사용.
* `mixup`: 두 클래스 사이의 결정 경계를 부드럽게 만드는 개념.
* `mixup_alpha`: 두 클래스를 얼마나 강하게 섞을 것인지 결정하는 개념

## 5. ResNet50 학습 (Transfer Learning)
```bash
python train.py \
  --data_dir ./data_3cls \
  --arch resnet50 \
  --freeze_backbone \
  --epochs 15 \
  --lr 1e-3 \
  --weighted_loss \
  --use_weighted_sampler
```

## 6. EfficientNetB0 학습 (Final Model)
```bash
python train.py \
  --data_dir ./data_3cls \
  --arch efficientnet_b0 \
  --epochs 15 \
  --lr 5e-4 \
  --unfreeze_at 5 \
  --weighted_loss \
  --use_weighted_sampler \
  --mixup \
  --mixup_alpha 0.2
```

## 7. 모델 평가 

### (1) RestNet50 평가
```bash
RUN=$(ls -dt outputs/resnet50_* | head -n1)

python eval.py \
  --data_dir ./data_3cls \
  --checkpoint "$RUN/best.ckpt" \
  --arch resnet50
```

### (2) EfficientNetB0 평가
```bash
RUN=$(ls -dt outputs/efficientnet_b0_* | head -n1)

python eval.py \
  --data_dir ./data_3cls \
  --checkpoint "$RUN/best.ckpt" \
  --arch efficientnet_b0
```

## 8. Grad-CAM 시각화
```bash
RUN=$(ls -dt outputs/efficientnet_b0_* | head -n1)

python grad_cam.py \
  --data_dir ./data_3cls \
  --checkpoint "$RUN/best.ckpt" \
  --arch efficientnet_b0 \
  --num_samples 6
```

## 9. OpenCV 기반 실도로 영상 테스트
```bash
RUN=$(ls -dt outputs/efficientnet_b0_* | head -n1)

python video_demo.py \
  --checkpoint "$RUN/best.ckpt" \
  --arch efficientnet_b0 \
  --class_names car truck bus \
  --video path/to/your_video.mp4
```