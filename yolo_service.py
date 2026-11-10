"""사진을 넣으면 YOLO 탐지 박스와 분할 마스크를 두 신뢰도 임계값으로 비교해 파일로 남긴다.

사용:
    python yolo_service.py input/bus.jpg input/내사진.jpg
    python yolo_service.py input/bus.jpg --conf 0.25 0.50

결과: outputs/yolo/ 아래에 사진별 결과 이미지, 박스 좌표 JSON, 마스크 PNG, 비교 이미지,
그리고 실행 기록 manifest.json과 이 코드의 사본.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DETECT_MODEL = "yolo26n.pt"
SEGMENT_MODEL = "yolo26n-seg.pt"
IMGSZ = 640
DEVICE = "cpu"
OUT = Path("outputs/yolo")
# 객체마다 다른 색(BGR). 마스크는 반투명으로 원본 위에 겹치고, 박스와 라벨도 같은 색을 쓴다.
PALETTE = [(56, 56, 255), (255, 157, 151), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (187, 212, 0), (255, 194, 0), (255, 115, 100), (236, 24, 0)]
MASK_ALPHA = 0.45


def color(i):
    return PALETTE[i % len(PALETTE)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_rows(result):
    """박스 하나 = 클래스·점수·좌표(원본 픽셀, 왼쪽 위 x1,y1 / 오른쪽 아래 x2,y2) 한 줄."""
    rows = []
    for i, b in enumerate(result.boxes):
        cls_id = int(b.cls)
        rows.append({
            "id": i,
            "class": result.names[cls_id],
            "class_id": cls_id,
            "conf": round(float(b.conf), 4),
            "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
        })
    return rows


def save_masks(result, rows, img, folder):
    """마스크를 원본 크기 PNG로 저장하고, 다시 열어 저장 전과 같은지 확인한다."""
    h, w = img.shape[:2]
    overlay = img.copy()
    folder.mkdir(parents=True, exist_ok=True)
    if result.masks is None:
        return overlay
    for row, poly in zip(rows, result.masks.xy):
        mask = np.zeros((h, w), np.uint8)
        if len(poly):
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        path = folder / f"{row['id']:02d}_{row['class'].replace(' ', '_')}.png"
        cv2.imwrite(str(path), mask)
        reloaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = row["xyxy"]
        box_area = max((x2 - x1) * (y2 - y1), 1.0)
        mask_area = int((mask > 0).sum())
        row.update({
            "mask_file": path.as_posix(),
            "mask_area_px": mask_area,
            "box_area_px": round(box_area),
            "background_in_box_ratio": round(1 - mask_area / box_area, 3),
            "mask_reload_same": bool(reloaded is not None and np.array_equal(reloaded, mask)),
        })
        on = mask > 0
        overlay[on] = (overlay[on] * (1 - MASK_ALPHA) + np.array(color(row["id"])) * MASK_ALPHA).astype(np.uint8)
    return overlay


def draw_boxes(img, rows):
    out = img.copy()
    for r in rows:
        x1, y1, x2, y2 = map(int, r["xyxy"])
        c = color(r["id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 3)
        text = f"#{r['id']} {r['class']} {r['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 8)  # 라벨이 사진 위로 잘리지 않게
        cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 6, ty), c, -1)
        cv2.putText(out, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def run_one(model, kind, img_path, img, conf, folder):
    """모델 하나 × 임계값 하나 → 결과 이미지 + JSON."""
    result = model.predict(str(img_path), imgsz=IMGSZ, device=DEVICE, conf=conf, verbose=False)[0]
    rows = box_rows(result)
    if kind == "segment":
        shown = draw_boxes(save_masks(result, rows, img, folder / "masks"), rows)
    else:
        shown = draw_boxes(img, rows)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{kind}.jpg"), shown)
    (folder / f"{kind}.json").write_text(
        json.dumps({"image": img_path.as_posix(), "model": DETECT_MODEL if kind == "detect" else SEGMENT_MODEL,
                    "conf": conf, "imgsz": IMGSZ, "device": DEVICE, "boxes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return rows, shown


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return out


def compare(rows_by_conf, low, high):
    """낮은 임계값에서는 있었지만 높은 임계값에서 사라진 박스(점수가 두 값 사이)."""
    gone = [r for r in rows_by_conf[low] if r["conf"] < high]
    return {"low": low, "high": high,
            "count_low": len(rows_by_conf[low]), "count_high": len(rows_by_conf[high]),
            "disappeared": [{k: r[k] for k in ("id", "class", "conf", "xyxy")} for r in gone]}


def process(img_path, models, confs):
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"사진을 열 수 없습니다: {img_path}")
    base = OUT / img_path.stem
    summary, panels = {}, [label(img, "original")]
    for kind, model in models.items():
        rows_by_conf = {}
        for conf in confs:
            rows, shown = run_one(model, kind, img_path, img, conf, base / f"conf{conf:.2f}")
            rows_by_conf[conf] = rows
            if kind == "detect":
                panels.append(label(shown, f"detect conf {conf:.2f} ({len(rows)} boxes)"))
        summary[kind] = compare(rows_by_conf, min(confs), max(confs))
    cv2.imwrite(str(base / "compare_detect.jpg"), np.hstack(panels))
    (base / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_manifest(images, confs, summaries):
    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python"] + sys.argv),
        "where": {"location": "로컬 PC (Windows, CPU만 사용)", "os": platform.platform(), "python": platform.python_version()},
        "settings": {"imgsz": IMGSZ, "device": DEVICE, "conf": confs},
        "models": {kind: {"file": name, "sha256": sha256(name) if Path(name).exists() else "미확인"}
                   for kind, name in (("detect", DETECT_MODEL), ("segment", SEGMENT_MODEL))},
        "packages": {p: version(p) for p in ("ultralytics", "torch", "opencv-python", "numpy")},
        "inputs": [{"file": p.as_posix(), "sha256": sha256(p), **source_of(p)} for p in images],
        "results": {p.stem: summaries[p.stem] for p in images},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def source_of(path):
    """input/sources.json에 적어 둔 출처와 공유 조건. 없으면 '미기록'으로 남긴다."""
    table = Path("input/sources.json")
    data = json.loads(table.read_text(encoding="utf-8")) if table.exists() else {}
    return data.get(path.name, {"source": "미기록", "license": "미기록"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--conf", nargs=2, type=float, default=[0.25, 0.50])
    args = ap.parse_args()
    confs = sorted(args.conf)
    OUT.mkdir(parents=True, exist_ok=True)
    models = {"detect": YOLO(DETECT_MODEL), "segment": YOLO(SEGMENT_MODEL)}
    summaries = {}
    for p in args.images:
        summaries[p.stem] = process(p, models, confs)
        s = summaries[p.stem]["detect"]
        print(f"{p.name}: detect {s['count_low']}개(conf {confs[0]}) → {s['count_high']}개(conf {confs[1]}), 사라진 박스 {len(s['disappeared'])}개")
    write_manifest(args.images, confs, summaries)
    (OUT / "code").mkdir(exist_ok=True)
    shutil.copy(__file__, OUT / "code" / Path(__file__).name)
    print(f"저장: {OUT.resolve()}")


if __name__ == "__main__":
    main()
