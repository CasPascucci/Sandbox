import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import random

# 1. Settings
CONFIDENCE_THRESHOLD = 0.25  # only show detections >=25%

# 2. COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 3. Unique colors for each class
random.seed(42)
COLORS = [
    tuple(random.randint(0, 255) for _ in range(3))
    for _ in range(len(COCO_INSTANCE_CATEGORY_NAMES))
]

# 4. Device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# 5. Load model and move to device
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device).eval()

# 6. Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 7. Preprocess: BGR→RGB, to tensor, normalize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

    # 8. Inference
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # 9. Post-process & draw
    boxes  = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()

    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = label.item()
        color = COLORS[cls_id]
        name  = COCO_INSTANCE_CATEGORY_NAMES[cls_id]
        text  = f"{name}: {score:.2f}"

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # draw label background
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        # draw text
        cv2.putText(frame, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 10. Display
    cv2.imshow("Torchvision GPU Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
