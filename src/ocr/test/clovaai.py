# clovaai.py

import os, sys, json
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
from json import JSONDecodeError
from torchvision.transforms import InterpolationMode

# CRAFT 소스 디렉터리를 sys.path에 추가
CRAFT_SRC = r"D:\Codes\Project\NODGU\CRAFT-pytorch"          # CRAFT 소스 루트
CRAFT_WEIGHT = r"D:\Codes\Project\NODGU\nodgu-data\craft_mlt_25k.pth"  # CRAFT 모델 가중치
RECOG_SRC = r"D:\Codes\Project\NODGU\deep-text-recognition-benchmark"
RECOG_WEIGHT = r"D:\Codes\Project\NODGU\nodgu-data\recognition_model.pth"
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

def _box_to_xyxy_scalar(b):
    """CRAFT box(리스트/ndarray/점 집합)를 [x1,y1,x2,y2] float로 정규화."""
    arr = np.asarray(b, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = map(float, arr.tolist())
    else:
        # (N,2) 형태 등 → AABB로 변환
        xs = arr[:, 0].astype(np.float32)
        ys = arr[:, 1].astype(np.float32)
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return x1, y1, x2, y2

def _sort_boxes(norm_boxes):
    """
    읽기 순서 정렬: '줄 단위 클러스터링 → 각 줄 내부 좌→우'
    norm_boxes: [(x1, y1, x2, y2), ...]
    return:     [(x1, y1, x2, y2), ...]  # 올바른 인덱스 순서
    """
    import numpy as np

    if not norm_boxes:
        return []

    # 중앙 y, 높이 계산
    items = []
    heights = []
    for x1, y1, x2, y2 in norm_boxes:
        cy = (y1 + y2) / 2.0
        h  = max(1.0, (y2 - y1))
        heights.append(h)
        items.append([float(x1), float(y1), float(x2), float(y2), float(cy)])

    h_med = float(np.median(heights))
    line_tol = max(12, int(h_med * 0.7))   # 필요하면 0.6~0.9 사이로 미세조정

    # 1) 라인 클러스터링(위→아래)
    lines = []  # list[list[item]]
    for it in sorted(items, key=lambda t: (t[1], t[0])):  # y1, x1 1차 정렬
        cy = it[4]
        placed = False
        for L in lines:
            # 현재 라인의 대표 y(중앙)과 비교
            L_cy = sum(p[4] for p in L) / len(L)
            if abs(cy - L_cy) <= line_tol:
                L.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    # 2) 라인 위→아래, 각 라인 내부 좌→우
    lines.sort(key=lambda L: sum(p[4] for p in L) / len(L))
    out = []
    for L in lines:
        L.sort(key=lambda p: p[0])  # x1
        out += [(p[0], p[1], p[2], p[3]) for p in L]

    return out



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

def craft_detect(net, image_rgb, device="cpu",
                 text_thresh=0.7, link_thresh=0.4, low_text=0.4,
                 canvas_size=1280, mag_ratio=1.5):
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

    # 박스만 반환 (poly=False)
    boxes, _ = craft_utils.getDetBoxes(score_text, score_link,
                                       text_thresh, link_thresh, low_text, poly=False)

    # 원본 좌표계로 복원
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes

##### text-recognition-benchmark #####

# 인식기 util
def _strip_module_prefix(state):
    from collections import OrderedDict
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    new = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    return new

from PIL import Image
from torchvision import transforms

def _get_transform(imgH, imgW, rgb=True):
    # Pillow 버전 호환용 resample 상수
    try:
        BICUBIC = Image.Resampling.BICUBIC   # Pillow>=9
    except AttributeError:
        BICUBIC = Image.BICUBIC              # Pillow<9

    class KeepRatioResizePad(object):
        def __init__(self, H, W, rgb):
            self.H, self.W, self.rgb = H, W, rgb
        def __call__(self, im: Image.Image):
            # 모델 입력 채널에 맞게 변환
            im = im.convert("RGB" if self.rgb else "L")

            w, h = im.size
            scale = min(self.W / max(1, w), self.H / max(1, h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            im_rs = im.resize((nw, nh), resample=BICUBIC)

            bg = Image.new("RGB" if self.rgb else "L",
                           (self.W, self.H),
                           color=(0, 0, 0) if self.rgb else 0)  # 학습과 동일: 검은 배경
            bg.paste(im_rs, ((self.W - nw) // 2, (self.H - nh) // 2))
            return bg

    return transforms.Compose([
        KeepRatioResizePad(imgH, imgW, rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * (3 if rgb else 1),
                             std=[0.5] * (3 if rgb else 1)),
    ])



# 체크포인트에서 아키텍처 자동 추론
def _infer_arch(state_no_module):
    keys = list(state_no_module.keys())
    has_tps  = any('GridGenerator' in k or 'LocalizationNetwork' in k for k in keys)
    is_resnet = any('FeatureExtraction.ConvNet.layer1.0' in k for k in keys)
    has_bilstm = any(k.startswith('SequenceModeling.') for k in keys)
    is_attn = any(k.startswith('Prediction.attention_cell') for k in keys)

    # 입력 채널 추정 (첫 conv weight shape: [out, in, k, k])
    in_ch = None
    for probe_key in ('FeatureExtraction.ConvNet.conv0_1.weight', 'FeatureExtraction.ConvNet.conv0.weight'):
        if probe_key in state_no_module:
            in_ch = state_no_module[probe_key].shape[1]
            break
    if in_ch is None:
        # 백업: 다른 키라도 찾아본다
        for k, v in state_no_module.items():
            if k.endswith('conv0_1.weight') and v.ndim == 4:
                in_ch = v.shape[1]
                break
    in_ch = 1 if in_ch == 1 else 3  # 기본 3

    # num_class 추정 (generator.weight [num_class, hidden])
    num_class = None
    for k in ('Prediction.generator.weight', ):
        if k in state_no_module:
            num_class = state_no_module[k].shape[0]
            break

    arch = {
        'Transformation': 'TPS' if has_tps else 'None',
        'FeatureExtraction': 'ResNet' if is_resnet else 'VGG',
        'SequenceModeling': 'BiLSTM' if has_bilstm else 'None',
        'Prediction': 'Attn' if is_attn else 'CTC',
        'input_channel': in_ch,
        'num_class_from_ckpt': num_class
    }
    return arch

# 인식기 loader
def load_textrec(device="cpu"):
    import types

    # 1) opt/charset 읽기
    opt = json.load(open(RECOG_OPT, "r", encoding="utf-8"))
    orig_character = opt["character"]              # 문자열(권장) 또는 문자열 유사 객체
    if not isinstance(orig_character, str):
        # 혹시 리스트로 저장돼 있으면 문자열로 환원
        orig_character = "".join(orig_character)
    raw_state = torch.load(RECOG_WEIGHT, map_location=device)

    # 2) 프리픽스 제거
    state = _strip_module_prefix(raw_state)

    # 3) ckpt에서 실측 치수 추출
    #   - LSTM Attn: weight_ih.shape = [4H, H+K], weight_hh.shape=[4H, H], generator.weight.shape=[K, hidden]
    try:
        w_ih = state["Prediction.attention_cell.rnn.weight_ih"]
        w_hh = state["Prediction.attention_cell.rnn.weight_hh"]
        hidden_size_ckpt = w_hh.shape[1]
        num_class_from_rnn = w_ih.shape[1] - hidden_size_ckpt
    except KeyError:
        w_ih = None
        hidden_size_ckpt = int(opt.get("hidden_size", 256))
        num_class_from_rnn = None

    try:
        num_class_from_gen = state["Prediction.generator.weight"].shape[0]
    except KeyError:
        num_class_from_gen = None

    # 최종 num_class는 ckpt 기준(가능하면)
    candidates = [c for c in [num_class_from_rnn, num_class_from_gen] if c is not None]
    if candidates:
        num_class_ckpt = min(candidates)  # 일반적으로 둘이 동일
    else:
        # 백업(권장X)
        is_attn_opt = opt.get("Prediction", "Attn").lower() != "ctc"
        num_class_ckpt = (len(orig_character) + 1) if is_attn_opt else len(orig_character)
        print("[WARN] ckpt에서 num_class를 유추 못해 opt 기반으로 설정:", num_class_ckpt)

    # 4) 아키텍처/입력채널 추론
    arch = _infer_arch(state)
    is_attn = (arch['Prediction'].lower() == 'attn')
    rgb = (arch['input_channel'] == 3)

    # 5) 특수토큰 갭 가설 k ∈ {1, 0, 2} 순차 시도
    #    k=1: 원 레포 규칙(len(charset)+1 == num_class)
    #    k=0: 특수토큰을 charset에 이미 포함시킨 포크
    #    k=2: charset에 [GO]와 [s]가 둘 다 포함된 포크
    def _build_converter(charset_str, attn=True, k_gap=1):
        # 목표 charset 길이
        target_len = max(1, num_class_ckpt - k_gap)
        ch = charset_str
        if len(ch) > target_len:
            ch = ch[:target_len]
        elif len(ch) < target_len:
            ch = ch + ("▢" * (target_len - len(ch)))  # 임시 채움(가능하면 실제 학습 charset 사용 권장)

        if attn:
            conv = AttnLabelConverter(ch)
            # 모델의 num_class는 ckpt 수치 고정(=num_class_ckpt)
            final_num_class = num_class_ckpt
        else:
            conv = CTCLabelConverter(ch)
            final_num_class = num_class_ckpt

        return ch, conv, final_num_class

    def _make_model_and_try(k_gap):
        # ModelOpt 구성
        ch, converter, final_num_class = _build_converter(orig_character, attn=is_attn, k_gap=k_gap)
        ModelOpt = types.SimpleNamespace(
            imgH=int(opt.get("imgH", 32)),
            imgW=int(opt.get("imgW", 100)),
            rgb=rgb,
            Transformation=arch['Transformation'],
            FeatureExtraction=arch['FeatureExtraction'],
            SequenceModeling=arch['SequenceModeling'],
            Prediction=arch['Prediction'],
            num_fiducial=int(opt.get("num_fiducial", 20)),
            input_channel=arch['input_channel'],
            output_channel=int(opt.get("output_channel", 512)),
            hidden_size=hidden_size_ckpt if hidden_size_ckpt is not None else int(opt.get("hidden_size", 256)),
            max_len=int(opt.get("max_len", 25)),
            num_class=final_num_class,                 # ← ckpt 기준 고정
            batch_max_length=int(opt.get("max_len", 25)),
        )
        print(f"[INFO] 시도(k={k_gap}) → Pred={ModelOpt.Prediction}, in_ch={ModelOpt.input_channel}, "
              f"hidden={ModelOpt.hidden_size}, num_class={ModelOpt.num_class}, charset_len={len(converter.character)}")

        model = Model(ModelOpt).to(device)
        try:
            # 먼저 strict=True 시도
            model.load_state_dict(state, strict=True)
            print(f"[OK] strict=True 로드 성공 (k={k_gap})")
            return model, converter, ModelOpt
        except Exception as e_true:
            print(f"[INFO] strict=True 실패 (k={k_gap}) → {e_true}")
            try:
                # strict=False 재시도 → 성공하면 사용 (헤드 일부 재초기화 필요할 수 있음)
                model.load_state_dict(state, strict=False)
                print(f"[OK] strict=False 로드 (k={k_gap})")
                return model, converter, ModelOpt
            except Exception as e_false:
                print(f"[NG] strict=False도 실패 (k={k_gap}) → {e_false}")
                return None, None, None

    # k 순서대로 시도
    tried = []
    for k_gap in (1, 0, 2):
        model, converter, ModelOpt = _make_model_and_try(k_gap)
        tried.append(k_gap)
        if model is not None:
            # 최종 성공
            tfm = _get_transform(ModelOpt.imgH, ModelOpt.imgW, ModelOpt.rgb)
            model.eval()
            return model, converter, ModelOpt, tfm

    # 모두 실패 시 마지막 시도 정보 출력 후 예외
    raise RuntimeError(f"Attn num_class/charset 조합 자동탐색 실패: 시도한 k={tried}, "
                       f"ckpt_num_class={num_class_ckpt}, hidden={hidden_size_ckpt}")
 

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
def detect_and_recognize(image_rgb: np.ndarray,
                         craft_model, textrec_pack, device="cpu",
                         text_thresh=0.7, link_thresh=0.4, low_text=0.4,
                         viz: bool = True, conf_thr: float = 0.0):

    H, W = image_rgb.shape[:2]
    boxes = craft_detect(
        craft_model, image_rgb, device=device,
        text_thresh=text_thresh, link_thresh=link_thresh, low_text=low_text
    )

    norm_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = _box_to_xyxy_scalar(b)
        # 클리핑 및 int 변환은 나중에
        norm_boxes.append((x1, y1, x2, y2))

    norm_boxes = _sort_boxes(norm_boxes)

    crops = []
    meta  = []
    for x1, y1, x2, y2 in norm_boxes:
        # ★ 클리핑 + int
        x1 = max(0, min(int(x1), W-1)); x2 = max(0, min(int(x2), W-1))
        y1 = max(0, min(int(y1), H-1)); y2 = max(0, min(int(y2), H-1))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_rgb[y1:y2, x1:x2, :]
        if crop.size == 0:
            continue
        crops.append(crop)
        meta.append({"bbox": [x1, y1, x2, y2]})

    if not crops:
        return [], image_rgb.copy()

    recs = recog_text(crops, model_pack=textrec_pack, device=device)
    results = []
    for m, r in zip(meta, recs):
        if r["conf"] >= conf_thr:
            results.append({
                "bbox": m["bbox"],
                "text": r["text"],
                "conf": float(r["conf"]),
            })

    viz_img = image_rgb.copy()
    if viz:
        for i, it in enumerate(results, 1):
            x1,y1,x2,y2 = it["bbox"]
            cv2.rectangle(viz_img, (x1,y1), (x2,y2), (255,0,0), 2)
            
            idx_text = str(i)
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness  = 1

            tx = x1 + 2
            ty = max(10, y1 - 4) 
            cv2.putText(viz_img, idx_text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(viz_img, idx_text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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