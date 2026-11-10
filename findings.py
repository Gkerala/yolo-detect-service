"""README의 '원본 대조에서 발견한 것'을 원본과 결과를 나란히 확대한 그림으로 만든다.

yolo_service.py를 먼저 실행한 뒤:
    python findings.py
결과: outputs/yolo/kitchen/findings/*.jpg
좌표는 outputs/yolo/kitchen/conf0.25/*.json의 xyxy(원본 픽셀)를 그대로 쓴다.
"""
from pathlib import Path

import cv2
import numpy as np

BASE = Path("outputs/yolo/kitchen")
ORIGINAL = Path("input/kitchen.jpg")
MARGIN = 60  # 박스 주변을 조금 더 보여 준다
HEIGHT = 360  # 나란히 놓을 때 맞출 높이

# (파일 이름, 원본 좌표 xyxy, [(제목, 결과 그림 경로)])
FINDINGS = [
    ("wallet_as_cell_phone", [1292, 853, 1411, 915],
     [("detect conf 0.25: #0 cell phone 0.68", BASE / "conf0.25/detect.jpg"),
      ("detect conf 0.50: still kept", BASE / "conf0.50/detect.jpg")]),
    ("plant_mask_edges", [16, 178, 518, 872],
     [("segment conf 0.25: #3 potted plant", BASE / "conf0.25/segment.jpg")]),
    ("bottle_0.4959", [1269, 663, 1325, 761],
     [("segment conf 0.25: bottle 0.4959", BASE / "conf0.25/segment.jpg"),
      ("segment conf 0.50: dropped", BASE / "conf0.50/segment.jpg")]),
    ("sofa_and_vase", [0, 540, 660, 1512],
     [("detect conf 0.25: chair / cup", BASE / "conf0.25/detect.jpg")]),
]


def crop(img, xyxy):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    return img[max(int(y1) - MARGIN, 0):min(int(y2) + MARGIN, h), max(int(x1) - MARGIN, 0):min(int(x2) + MARGIN, w)]


def titled(img, text):
    scale = HEIGHT / img.shape[0]
    img = cv2.resize(img, (max(int(img.shape[1] * scale), 1), HEIGHT), interpolation=cv2.INTER_CUBIC)
    bar = np.full((34, max(img.shape[1], 360), 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    body = np.full((HEIGHT, bar.shape[1], 3), 255, np.uint8)
    body[:, :img.shape[1]] = img
    return np.vstack([bar, body])


def main():
    original = cv2.imread(str(ORIGINAL))
    if original is None:
        raise SystemExit(f"원본을 열 수 없습니다: {ORIGINAL}")
    out = BASE / "findings"
    out.mkdir(parents=True, exist_ok=True)
    for name, xyxy, shown in FINDINGS:
        panels = [titled(crop(original, xyxy), "original")]
        for title, path in shown:
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"결과 그림이 없습니다. yolo_service.py를 먼저 실행하세요: {path}")
            panels.append(titled(crop(img, xyxy), title))
        gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.hstack([row, gap, p])
        cv2.imwrite(str(out / f"{name}.jpg"), row)
        print("저장:", out / f"{name}.jpg")


if __name__ == "__main__":
    main()
