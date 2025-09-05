import os, sys, json
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
from json import JSONDecodeError

# CRAFT 소스 디렉터리를 sys.path에 추가
CRAFT_SRC = r"D:\Codes\Project\NODGU\CRAFT-pytorch"          # CRAFT 소스 루트
CRAFT_WEIGHT = r"D:\Codes\Project\NODGU\nodgu-data\craft_mlt_25k.pth"  # CRAFT 모델 가중치
RECOG_SRC = r"D:\Codes\Project\NODGU\deep-text-recognition-benchmark"
RECOG_WEIGHT = r"D:\Codes\Project\NODGU\nodgu-data\recognition_model_final.pth"
RECOG_OPT    = r"D:\Codes\Project\NODGU\nodgu-data\opt.json"

API_URL = "https://api.nodgu.shop/api/v1/notice/notice/noOcrData"
response = requests.get(API_URL, timeout=20)

SAVE_VIZ = r"D:\Codes\Project\NODGU\nodgu-data\output"
os.makedirs(SAVE_VIZ, exist_ok=True)

sys.path.insert(0, CRAFT_SRC)
sys.path.insert(0, RECOG_SRC)

from craft import CRAFT
import craft_utils
import imgproc

from model import Model
from utils import AttnLabelConverter, CTCLabelConverter
from torchvision import transforms


##### 유틸 #####
def _download_image(url: str) -> np.ndarray:
    """URL에서 이미지를 받아 RGB NumPy(H,W,3)로 반환."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return np.array(img)  # RGB

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """사각형 4점을 좌상-우상-우하-좌하 순으로 정렬."""
    # pts: (4,2)
    x_sum = pts.sum(1)
    x_diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(x_sum)]
    br = pts[np.argmax(x_sum)]
    tr = pts[np.argmin(x_diff)]
    bl = pts[np.argmax(x_diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _warp_quad(rgb: np.ndarray, quad: np.ndarray, pad: int = 2) -> np.ndarray:
    """폴리곤(사각형 가정)을 직사각형으로 퍼스펙티브 보정해서 잘라냄."""
    quad = _order_quad(quad.astype(np.float32))
    w = int(np.linalg.norm(quad[1] - quad[0]))
    h = int(np.linalg.norm(quad[3] - quad[0]))
    w = max(w, 8)
    h = max(h, 8)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    crop = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_CUBIC)
    # 여백 패딩(경계 손실 방지)
    if pad > 0:
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    return crop

def _bbox_from_poly(poly):
    """폴리곤에서 AABB(bbox) [x1,y1,x2,y2]"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def _sort_boxes_tblr(polys):
    """텍스트 읽기 순서 정렬(위→아래, 좌→우). polys: list of 4점(또는 n점)"""
    def key(p):
        arr = np.array(p)
        y = arr[:,1].mean()
        x = arr[:,0].min()
        return (int(round(y/10))*10, x)  # y를 버킷으로 묶어 줄바꿈 취급
    return sorted(polys, key=key)


##### CRAFT #####
def load_craft(device="cpu"):
    model = CRAFT()
    state = torch.load(CRAFT_WEIGHT, map_location=device)

    if any(k.startswith("module.") for k in state.keys()):
        from collections import OrderedDict
        new_state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
        state = new_state
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def craft_detect(net, image_rgb, device="cpu", text_thresh=0.7, link_thresh=0.4, low_text=0.4, canvas_size=1280, mag_ratio=1.5):
    # PIL/np -> CRAFT 입력 전처리
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image_rgb, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # 박스/폴리곤 생성
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_thresh, link_thresh, low_text, True)
    # 원본 좌표계로 복원
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
   
    # polys가 None인 항목 fallback
    if polys is None:
        polys = [None]*len(boxes)

    fixed_polys = []
    for b, p in zip(boxes, polys):
        if p is None:
            p = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.float32)
        fixed_polys.append(p)
    return boxes, fixed_polys


##### text-recognition-benchmark #####

# 인식기 util
def _strip_module_prefix(state):
    from collections import OrderedDict
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    new = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    return new

def _get_transform(imgH, imgW, rgb=True):
    from PIL import Image
    return transforms.Compose([
        transforms.Resize((imgH, imgW), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*(3 if rgb else 1),
                             std=[0.5]*(3 if rgb else 1)),
    ])

# 인식기 loader
def load_textrec(device="cpu"):
    import types
    # opt/character 로드
    opt = json.load(open(RECOG_OPT, "r", encoding="utf-8"))
    character = opt["character"]               # opt.json에서 읽음
    is_attn = opt.get("Prediction", "Attn").lower() != "ctc"
    # 모델이 기대하는 클래스 수
    num_class_from_opt = (len(character) + 1) if is_attn else len(character)

    # converter & num_class
    if opt.get("Prediction", "Attn").lower() == "ctc":
        converter = CTCLabelConverter(character)
        num_class = len(converter.character)
    else:
        converter = AttnLabelConverter(character)
        num_class = len(converter.character) + 1  # [EOS]

    state = torch.load(RECOG_WEIGHT, map_location=device)

    try:
        ckpt_num_class = state["Prediction.generator.weight"].shape[0]
    except KeyError:
        # 혹은 다른 키로 저장된 경우를 대비
        for k, v in state.items():
            if k.endswith("Prediction.generator.weight"):
                ckpt_num_class = v.shape[0]
                break
        else:
            ckpt_num_class = None

    # ★ ckpt와 불일치하면 ckpt 기준으로 옵션 보정
    if ckpt_num_class is not None and ckpt_num_class != num_class_from_opt:
        print(f"[WARN] checkpoint num_class={ckpt_num_class}, opt/charset num_class={num_class_from_opt}. "
            f"모델을 checkpoint 기준으로 보정합니다.")
        if is_attn:
            # EOS 포함 ⇒ character 길이는 ckpt_num_class - 1 이어야 함
            target_len = ckpt_num_class - 1
        else:
            target_len = ckpt_num_class

        # 길이를 맞춤 (순서 불일치 시 성능 저하 가능)
        if len(character) > target_len:
            character = character[:target_len]
        elif len(character) < target_len:
            character = character + ("▢" * (target_len - len(character)))  # 더미 채움 (권장X)

    # 이후 converter, ModelOpt 생성 시 num_class는 character 길이에 맞춰 설정
    if is_attn:
        converter = AttnLabelConverter(character)
        num_class = len(converter.character) + 1
    else:
        converter = CTCLabelConverter(character)
        num_class = len(converter.character)
        
    rgb = bool(opt.get("rgb", True))
    input_channel = int(opt.get("input_channel", 3 if rgb else 1))
    # conv 첫 레이어 키로 판단
    first_conv_key = "FeatureExtraction.ConvNet.conv0_1.weight"
    if first_conv_key in state and state[first_conv_key].shape[1] == 1:
        rgb = False
        input_channel = 1

    # ModelOpt에 반영
    ModelOpt = types.SimpleNamespace(
        imgH=int(opt.get("imgH", 32)),
        imgW=int(opt.get("imgW", 100)),
        rgb=rgb,
        Transformation=opt.get("Transformation", "TPS"),
        FeatureExtraction=opt.get("FeatureExtraction", "ResNet"),
        SequenceModeling=opt.get("SequenceModeling", "BiLSTM"),
        Prediction=opt.get("Prediction", "Attn"),
        num_fiducial=int(opt.get("num_fiducial", 20)),
        input_channel=input_channel,               # ← 보정된 채널 수
        output_channel=int(opt.get("output_channel", 512)),
        hidden_size=int(opt.get("hidden_size", 256)),
        max_len=int(opt.get("max_len", 25)),
        num_class=num_class,
        batch_max_length=int(opt.get("max_len", 25)),
    )

    model = Model(ModelOpt)
    model.load_state_dict(state, strict=True)
    
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(_strip_module_prefix(state), strict=True)

    model.to(device).eval()
    tfm = _get_transform(ModelOpt.imgH, ModelOpt.imgW, ModelOpt.rgb)
    return model, converter, ModelOpt, tfm

# 인식 함수(one, batch)
@torch.no_grad()
def recog_text(crops, model_pack=None, device="cpu"):
    """
    crops: PIL.Image 또는 (H,W,3) numpy 배열 리스트
    return: [{"text": str, "conf": float}]
    """
    from PIL import Image
    model, converter, mopt, tfm = model_pack or load_textrec(device=device)

    # 배치 구성
    batch_tensors = []
    for im in crops:
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        elif isinstance(im, Image.Image):
            pass
        else:
            raise TypeError("crops 요소는 PIL.Image 또는 np.ndarray 여야 합니다.")
        batch_tensors.append(tfm(im))
    images = torch.stack(batch_tensors, dim=0).to(device)

    # CTC/Attn 분기
    if mopt.Prediction.lower() == "ctc":
        preds = model(images, text=None).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([preds.size(1)] * images.size(0))
        texts = converter.decode(preds_index.data, preds_size.data)
        confs = [float(torch.exp(p.max(1).values).mean().item()) for p in preds]  # 대략치
    else:
        import torch.nn.functional as F
        max_len = getattr(mopt, "max_len", 25)
        length_for_pred = torch.IntTensor([max_len] * images.size(0)).to(device)
        text_for_pred = torch.LongTensor(images.size(0), max_len+1).fill_(0).to(device)
        preds = model(images, text_for_pred, is_train=False)
        probs = F.softmax(preds, dim=2)
        max_prob, preds_idx = probs.max(2)
        pred_str = converter.decode(preds_idx, length_for_pred)
        texts, confs = [], []
        for s, mp in zip(pred_str, max_prob):
            eos = s.find("[s]")
            s_clean = s[:eos] if eos != -1 else s
            texts.append(s_clean)
            L = len(s_clean)
            confs.append(float(mp[:L].mean().item()) if L > 0 else 0.0)

    return [{"text": t, "conf": c} for t, c in zip(texts, confs)]


##### 파이프라인: 감지 → 크롭 → 인식 → 시각화 #####
def detect_and_recognize(image_rgb: np.ndarray, craft_model, textrec_pack, device="cpu",
                         text_thresh=0.7, link_thresh=0.4, low_text=0.4,
                         viz: bool = True, conf_thr: float = 0.0):
    """
    반환:
    results = [
        {"poly": [[x,y],...4개], "bbox":[x1,y1,x2,y2], "text": str, "conf": float}
    ], viz_img(RGB, np.ndarray)
    """
    H, W = image_rgb.shape[:2]
    boxes, polys = craft_detect(
        craft_model, image_rgb, device=device,
        text_thresh=text_thresh, link_thresh=link_thresh, low_text=low_text
    )

    # 정렬(위→아래, 좌→우)
    polys = _sort_boxes_tblr(polys)

    # 폴리곤 기준 크롭(가능하면 퍼스펙티브 보정, 아니면 AABB)
    crops = []
    meta  = []
    for p in polys:
        p_np = np.array(p, dtype=np.float32)
        if len(p_np) >= 4:
            quad = p_np[:4]  # CRAFT는 보통 4점
            crop = _warp_quad(image_rgb, quad, pad=2)
            bbox = _bbox_from_poly(p)
        else:
            # fallback: 사각
            bbox = _bbox_from_poly(p)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            crop = image_rgb[y1:y2, x1:x2, :]
        if crop.size == 0:
            continue
        crops.append(crop)
        meta.append({"poly": p, "bbox": _bbox_from_poly(p)})

    if not crops:
        return [], image_rgb.copy()

    # 배치 인식
    recs = recog_text(crops, model_pack=textrec_pack, device=device)
    results = []
    for m, r in zip(meta, recs):
        if r["conf"] >= conf_thr:
            results.append({
                "poly": [list(map(float, pt)) for pt in m["poly"]],
                "bbox": list(map(int, m["bbox"])),
                "text": r["text"],
                "conf": float(r["conf"]),
            })

    # 시각화(선택)
    viz_img = image_rgb.copy()
    if viz:
        for it in results:
            poly = np.array(it["poly"], dtype=np.int32)
            cv2.polylines(viz_img, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
            x1,y1,x2,y2 = it["bbox"]
            label = f'{it["text"]} ({it["conf"]:.2f})'
            # 텍스트 배경
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(viz_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,0,0), -1)
            cv2.putText(viz_img, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    return results, viz_img

def main():
    data = []
    try:
        response = requests.get(API_URL, timeout=20)
        response.raise_for_status()

        if response.text:
            responseJSON = response.json()
            data = responseJSON['data']

        else:
            print("요청 성공. 응답 본문 비어있음")

    except requests.exceptions.HTTPError as e:
        # 200번대가 아닐 때
        print(f"HTTP 오류: {e}")
    except JSONDecodeError as e:
        # 응답 본문이 JSON 형식이 아닐 때
        print(f"JSON 디코딩 오류: {e}. 응답 텍스트: '{response.text}'")
    except requests.exceptions.RequestException as e:
        # 연결 문제와 같은 기타 requests
        print(f"요청 오류: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        craft_model = load_craft(device=device)
        textrec_pack = load_textrec(device=device)
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    for item in data:
        imgs = item['imgs']
        # print("새 이미지 리스트")
        for url in imgs:
            # print(f"\n[IMG] {url}")
            try:
                rgb = _download_image(url)  # RGB
            except Exception as e:
                print(f"이미지 다운로드 실패: {e}")
                continue
            try:
                results, viz_img = detect_and_recognize(
                    rgb, craft_model, textrec_pack, device=device,
                    text_thresh=0.7, link_thresh=0.4, low_text=0.4,
                    viz=True, conf_thr=0.0
                )

                lines = []

                # 결과 출력(JSON 라인)
                for i, r in enumerate(results, 1):
                    line = json.dumps({
                        "idx": i,
                        "text": r["text"]
                    }, ensure_ascii=False)
                    print(line)
                    lines.append(line)

                # 결과를 txt 파일로 저장
                if SAVE_VIZ:
                    out_txt = os.path.join(SAVE_VIZ, f"result_{os.path.basename(url).split('?')[0]}.txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    print(f"[SAVE] {out_txt}")

                # 시각화 저장
                if SAVE_VIZ:
                    # RGB→BGR 저장
                    out_name = os.path.join(SAVE_VIZ, f"viz_{os.path.basename(url).split('?')[0]}")
                    out_name = os.path.splitext(out_name)[0] + ".jpg"
                    cv2.imwrite(out_name, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
                    print(f"[SAVE] {out_name}")

            except Exception as e:
                print(f"처리 실패: {e}")

if __name__ == "__main__":
    main()