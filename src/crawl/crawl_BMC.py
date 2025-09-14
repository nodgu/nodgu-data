import os, csv, re, time, base64, requests
import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import BytesIO
from urllib.parse import urljoin
from PIL import Image
import numpy as np
from collections.abc import Mapping
from paddleocr import PaddleOCR
import unicodedata
from typing import List, Tuple, Dict, Union, Iterable

# 상수
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BASE = "https://bmcdorm.dongguk.edu"
LIST_URL = BASE + "/article/menu/list?pageIndex={}"
DETAIL_HREF_RE = re.compile(r"/article/menu/detail/(\d+)")
OUT_DIR = "output_BMC"; os.makedirs(OUT_DIR, exist_ok=True)
OP_CSV = os.path.join(OUT_DIR, "BMC.csv")

WEEK_KOR = ["월","화","수","목","금"]
SLOT_KWS = ["조식","중식","석식","코너A","코너B","코너a","코너b"] 

# ===== 디버그 =====
DEBUG = True
DBG_PREVIEW_N = 8
DBG_CELL_LINES_N = 12
def dbg(*args):
    if DEBUG: print("[DBG]", *args)
def wrn(*args): print("[WRN]", *args)

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def texts_from_paddle(result: Union[list, tuple]) -> list:
    out = []
    def push(txt):
        if isinstance(txt, str):
            t = normalize_text(txt)
            if t:
                out.append(t)
    def walk(x):
        if isinstance(x, dict):
            for key in ("text", "rec_text", "transcription", "label"):
                if key in x and isinstance(x[key], str):
                    push(x[key])
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            if len(x) >= 2 and isinstance(x[1], (list, tuple)):
                cand = x[1]
                if len(cand) >= 1 and isinstance(cand[0], str):
                    push(cand[0])
            for y in x:
                walk(y)
            return
        if isinstance(x, str):
            push(x)
    walk(result)
    dbg("texts_from_paddle ->", {"count": len(out), "sample": out[:10]})
    return out

def is_price_like(s: str) -> bool:
    s_norm = normalize_text(s)
    ok = ("원" in s_norm) or bool(re.search(r"\b[0-9]{1,3}(?:,[0-9]{3})+\b", s_norm))
    dbg("is_price_like <-", s, "=>", ok)
    return ok

def _binarize_for_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, -10)
    dbg("_binarize_for_lines ->", {"shape": th.shape, "nonzero": int(np.count_nonzero(th))})
    return th

def _detect_grid_line_maps(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = bin_img.shape[:2]
    hk = max(15, W // 60)
    vk = max(15, H // 45)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    h_tmp = cv2.erode(bin_img, h_kernel, iterations=1)
    h_lines = cv2.dilate(h_tmp, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    v_tmp = cv2.erode(bin_img, v_kernel, iterations=1)
    v_lines = cv2.dilate(v_tmp, v_kernel, iterations=1)

    h_lines = cv2.dilate(h_lines, np.ones((3,3),np.uint8), iterations=1)
    v_lines = cv2.dilate(v_lines, np.ones((3,3),np.uint8), iterations=1)

    dbg("_detect_grid_line_maps ->", {
        "hk": hk, "vk": vk,
        "h_nonzero": int(np.count_nonzero(h_lines)),
        "v_nonzero": int(np.count_nonzero(v_lines)),
        "shape": (H, W)
    })
    return h_lines, v_lines

def _merge_close_positions(positions: List[int], gap: int) -> List[int]:
    if not positions:
        dbg("_merge_close_positions -> empty (gap", gap, ")")
        return []
    positions = sorted(positions)
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] <= gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    dbg("_merge_close_positions ->", {"in": len(positions), "out": len(merged), "gap": gap, "sample": merged[:10]})
    return merged

def _extract_line_positions(h_lines: np.ndarray, v_lines: np.ndarray) -> Tuple[List[int], List[int]]:
    h_proj = (h_lines > 0).sum(axis=1)
    v_proj = (v_lines > 0).sum(axis=0)
    H = len(h_proj); W = len(v_proj)
    h_thresh = max(40, int(0.25 * W))
    v_thresh = max(40, int(0.20 * H))
    h_pos_raw = [i for i, v in enumerate(h_proj) if v > h_thresh]
    v_pos_raw = [i for i, v in enumerate(v_proj) if v > v_thresh]
    h_pos = _merge_close_positions(h_pos_raw, gap=max(6, H // 150))
    v_pos = _merge_close_positions(v_pos_raw, gap=max(6, W // 150))
    dbg("_extract_line_positions ->", {
        "h_thresh": h_thresh, "v_thresh": v_thresh,
        "h_count": len(h_pos), "v_count": len(v_pos),
        "h_sample": h_pos[:10], "v_sample": v_pos[:10]
    })
    return h_pos, v_pos


def detect_table_grid(img_bgr: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    bin_img = _binarize_for_lines(img_bgr)
    h_map, v_map = _detect_grid_line_maps(bin_img)
    h_pos, v_pos = _extract_line_positions(h_map, v_map)

    if h_pos and (h_pos[0] > 5):
        h_pos = [0] + h_pos
    if h_pos and (h_pos[-1] < img_bgr.shape[0]-6):
        h_pos = h_pos + [img_bgr.shape[0]-1]
    if v_pos and (v_pos[0] > 5):
        v_pos = [0] + v_pos
    if v_pos and (v_pos[-1] < img_bgr.shape[1]-6):
        v_pos = v_pos + [img_bgr.shape[1]-1]

    h_pos = sorted(set(h_pos))
    v_pos = sorted(set(v_pos))
    dbg("detect_table_grid ->", {
        "h_len": len(h_pos), "v_len": len(v_pos),
        "h_bounds": (h_pos[0], h_pos[-1]) if h_pos else None,
        "v_bounds": (v_pos[0], v_pos[-1]) if v_pos else None
    })
    return h_pos, v_pos, h_map, v_map

def build_grid_cells(h_pos: List[int], v_pos: List[int]) -> List[Dict]:
    cells = []
    for r in range(len(h_pos)-1):
        y1, y2 = h_pos[r], h_pos[r+1]
        if y2 - y1 < 12:
            continue
        for c in range(len(v_pos)-1):
            x1, x2 = v_pos[c], v_pos[c+1]
            if x2 - x1 < 12:
                continue
            cells.append({"row": r, "col": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    dbg("build_grid_cells ->", {"count": len(cells), "sample": cells[:5]})
    return cells

def draw_thick_grid_overlay(img_bgr: np.ndarray, h_pos: List[int], v_pos: List[int], out_path: str) -> None:
    out = img_bgr.copy()
    thick = max(3, min(img_bgr.shape[:2]) // 250)
    for x in v_pos:
        cv2.line(out, (x, 0), (x, img_bgr.shape[0]-1), (0,0,0), thickness=thick*2)
    for y in h_pos:
        cv2.line(out, (0, y), (img_bgr.shape[1]-1, y), (0,0,0), thickness=thick*2)
    cv2.imwrite(out_path, out)
    dbg("draw_thick_grid_overlay ->", out_path)

def draw_cell_boxes(img_bgr: np.ndarray, cells: List[Dict], out_path: str) -> None:
    out = img_bgr.copy()
    for c in cells:
        cv2.rectangle(out, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0,0,255), thickness=2)
    cv2.imwrite(out_path, out)
    dbg("draw_cell_boxes ->", out_path, "(cells:", len(cells), ")")

def save_cell_crops(img_bgr: np.ndarray, cells: List[Dict], save_dir: str, prefix: str = "cell") -> None:
    os.makedirs(save_dir, exist_ok=True)
    for c in cells:
        crop = img_bgr[c["y1"]:c["y2"], c["x1"]:c["x2"]].copy()
        fname = f"{prefix}_r{c['row']:02d}_c{c['col']:02d}.png"
        cv2.imwrite(os.path.join(save_dir, fname), crop)
    dbg("save_cell_crops ->", {"dir": save_dir, "count": len(cells)})


# ===== OCR =====
_OCR = None
def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(lang="korean")
        dbg("PaddleOCR 초기화 완료 (lang=korean)")
    return _OCR

def _ensure_c_contiguous(arr: np.ndarray) -> np.ndarray:
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr

def run_ocr(img_input: Union[bytes, bytearray, np.ndarray]):
    if isinstance(img_input, (bytes, bytearray)):
        img = Image.open(BytesIO(img_input)).convert("RGB")
        np_img_rgb = np.array(img)
    elif isinstance(img_input, np.ndarray):
        arr = _ensure_c_contiguous(img_input)
        if arr.ndim == 2:
            np_img_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            np_img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"지원하지 않는 배열 형태: shape={arr.shape}")
    else:
        raise TypeError("run_ocr는 bytes/bytearray 또는 np.ndarray만 받습니다.")

    dbg("OCR 입력:", {"shape": np_img_rgb.shape, "dtype": str(np_img_rgb.dtype), "contiguous": bool(np_img_rgb.flags['C_CONTIGUOUS'])})
    res = get_ocr().predict(np_img_rgb)

    texts = texts_from_paddle(res)
    dbg("run_ocr ->", {"texts_count": len(texts), "sample": texts[:5]})
    return res, np_img_rgb

# =========================
def cut_header_by_grid(img_bgr: np.ndarray,
                       h_pos: list[int],
                       v_pos: list[int]) -> np.ndarray:
    """
    표 바깥 여백/타이틀(‘누리터 식당 주간 메뉴표’) 제거.
    - 그리드의 첫 내부 수평선~마지막 내부 수평선 사이만 남김
    - OCR 사용 안 함 (기능 보존)
    """
    if len(h_pos) < 3 or len(v_pos) < 2:
        return img_bgr
    # 외곽 프레임(0, H-1)을 제외한 '내부 라인'만 사용
    top    = h_pos[1]               # 첫 내부 가로선
    bottom = h_pos[-2]              # 마지막 내부 가로선
    left   = v_pos[1] if len(v_pos) > 2 else v_pos[0]
    right  = v_pos[-2] if len(v_pos) > 2 else v_pos[-1]

    # 안전 패딩(라인 두께 오차 흡수)
    pad = max(2, (bottom-top)//400)
    y0, y1 = max(0, top+pad), min(img_bgr.shape[0]-1, bottom-pad)
    x0, x1 = max(0, left+pad), min(img_bgr.shape[1]-1, right-pad)
    return img_bgr[y0:y1, x0:x1]

def cut_footer_by_green_label(img_bgr: np.ndarray,
                              h_pos: list[int],
                              v_pos: list[int]) -> np.ndarray:
    """
    좌측 첫 열의 연두색 '원산지표시' 행 탐지 → 그 윗선에서 아래를 통째로 제거.
    OCR 사용 안 함.
    """
    if len(h_pos) < 3 or len(v_pos) < 2:
        return img_bgr

    x1, x2 = v_pos[0], v_pos[1]  # 좌측 라벨열
    cut_y = None
    for r in range(len(h_pos)-1):
        y1, y2 = h_pos[r], h_pos[r+1]
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 40, 70), (90, 255, 255))  # 연두색 대역
        if (mask.mean()/255.0) > 0.20:  # 필요하면 0.15~0.30 사이로 튜닝
            cut_y = y1
            break
    if cut_y is None:
        return img_bgr
    return img_bgr[:max(0, cut_y-2), :]


def ocr_lines_from_result(result) -> List[str]:
    """
    PaddleOCR predict 결과에서 (box, text)를 뽑아 y중심으로 라인 그룹핑.
    같은 줄은 좌->우 정렬 후 공백으로 붙여 한 문자열로 반환.
    """
    boxes_texts = []

    def walk(x):
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            box, rec = x[0], x[1]
            # box: 점 4개 형태 추정, rec: (text, conf) 형태
            if (isinstance(box, (list, tuple)) and len(box) >= 4 and
                isinstance(rec, (list, tuple)) and len(rec) >= 1 and
                isinstance(rec[0], str)):
                txt = normalize_text(rec[0])
                if txt:
                    try:
                        ys = [pt[1] for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                        xs = [pt[0] for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                        if ys and xs:
                            y_c = sum(ys) / len(ys)
                            x_c = sum(xs) / len(xs)
                            boxes_texts.append((y_c, x_c, txt))
                    except Exception:
                        pass
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(result)

    if not boxes_texts:
        return []

    # y중심으로 정렬, 근접 y는 같은 라인으로 묶기
    boxes_texts.sort(key=lambda t: (t[0], t[1]))
    lines = []
    current = []
    y_thr = None
    for y, x, txt in boxes_texts:
        if not current:
            current = [(y, x, txt)]
            y_thr = y
            continue
        if abs(y - y_thr) <= 12:  # 같은 줄로 간주(픽셀 기준; 필요시 미세조정)
            current.append((y, x, txt))
        else:
            current.sort(key=lambda t: t[1])
            lines.append(" ".join(t[2] for t in current))
            current = [(y, x, txt)]
            y_thr = y
    if current:
        current.sort(key=lambda t: t[1])
        lines.append(" ".join(t[2] for t in current))

    return [normalize_text(s) for s in lines if s.strip()]


def apply_grid_and_debug(np_img: np.ndarray, post_id: str, out_dir: str) -> List[Dict]:
    # 1) 1차 그리드
    h_pos_full, v_pos_full, _, _ = detect_table_grid(np_img)
    if not h_pos_full or not v_pos_full:
        wrn("격자 검출 실패"); return []

    # 1-1) 상단 타이틀 컷 (여기 추가)
    img_cut_top = cut_header_by_grid(np_img, h_pos_full, v_pos_full)

    # 1-2) 하단 원산지/공지 컷 (여기 추가)
    #  → 컷했으니 다시 한 번 그리드(대략) 잡고 그 정보로 자른다
    h_pos_tmp, v_pos_tmp, _, _ = detect_table_grid(img_cut_top)
    img_main = cut_footer_by_green_label(img_cut_top, h_pos_tmp, v_pos_tmp)

    # 2) 최종 그리드 재검출 (여기부터는 기존 코드 유지)
    h_pos, v_pos, _, _ = detect_table_grid(img_main)
    cells = build_grid_cells(h_pos, v_pos)

    # 3) 요일/카테고리 추정
    rows = sorted(set(c["row"] for c in cells))
    cols = sorted(set(c["col"] for c in cells))
    min_row, min_col = (min(rows) if rows else 0), (min(cols) if cols else 0)

    # (A) 열→요일 (상단 2행)
    header_rows = 2
    top_rows = [r for r in rows if r < (min_row + header_rows)]
    col2day: Dict[int, str] = {}
    for cidx in cols:
        cand = []
        for r in top_rows:
            for c in cells:
                if c["row"] == r and c["col"] == cidx:
                    crop = img_main[c["y1"]:c["y2"], c["x1"]:c["x2"]]
                    r_out, _ = run_ocr(crop)
                    cand += texts_from_paddle(r_out)
        joined = " ".join(cand)
        day = "?"
        m = re.search(r"([월화수목금])\s*요일?", joined) or \
            re.search(r"\(([월화수목금])\)", joined) or \
            re.search(r"\d{1,2}\s*[./-]\s*\d{1,2}\s*\(([월화수목금])\)", joined)
        if m:
            day = m.group(1)
        col2day[cidx] = day
    dbg("apply: col2day ->", col2day)

    # 위치 보정(월~금 고정)
    ordered_cols = sorted(cols)
    anchors = [(ordered_cols.index(c), WEEK_KOR.index(d))
               for c,d in ((c, col2day.get(c)) for c in ordered_cols) if d in WEEK_KOR]
    if anchors:
        base = max(0, min(min(i-d for i,d in anchors), len(ordered_cols)-len(WEEK_KOR)))
        for k, day_name in enumerate(WEEK_KOR):
            idx = base + k
            if 0 <= idx < len(ordered_cols):
                col2day[ordered_cols[idx]] = day_name
    else:
        last5 = ordered_cols[-len(WEEK_KOR):]
        for k, cidx in enumerate(last5):
            col2day[cidx] = WEEK_KOR[k]
    dbg("apply: col2day(after-pos-fix) ->", col2day)

    # (B) 행→슬롯 (좌측 2열)
    left_cols_to_scan = 2
    row2slot: Dict[int, str] = {}
    SLOT_SCAN = ("조식","코너A","코너B","석식","코너a","코너b")

    for ridx in rows:
        cand = []
        for c in cells:
            if c["row"] == ridx and c["col"] < (min_col + left_cols_to_scan):
                crop = img_main[c["y1"]:c["y2"], c["x1"]:c["x2"]]
                r_out, _ = run_ocr(crop)
                cand += texts_from_paddle(r_out)
        joined = " ".join(cand)
        slot = "?"
        for kw in SLOT_SCAN:
            if re.search(kw, joined, flags=re.I):
                slot = "코너A" if kw.lower()=="코너a" else ("코너B" if kw.lower()=="코너b" else kw)
                break
        row2slot[ridx] = slot

    # 슬롯 Forward-Fill (조식/코너A/코너B/석식만 전파)
    dbg("apply: row2slot(before-ffill) ->", row2slot)
    current_slot = None
    FFILL_SLOTS = {"조식","코너A","코너B","석식"}
    for ridx in sorted(rows):
        v = row2slot.get(ridx, "?")
        if v in FFILL_SLOTS:
            current_slot = v
        elif v == "?" and current_slot:
            row2slot[ridx] = current_slot
    dbg("apply: row2slot(after-ffill) ->", row2slot)


    # 4) 열(요일) 기준 출력 (라인 묶음)
    # (D) 열(요일) 기준 출력 — 슬롯별로 '메뉴 → 가격' 순서 보장
    BLOCK_KWS = ("누리터","주간","메뉴표")
    any_printed = False

    day_cols = [cidx for cidx in cols if col2day.get(cidx, "?") in WEEK_KOR]
    day_cols.sort(key=lambda x: WEEK_KOR.index(col2day.get(x)))  # 월→금

    # --- 가격 행 탐지 (요일 컬럼만 기준) ---
    row_is_price: Dict[int, bool] = {}
    for ridx in rows:
        tokens = []
        for c in cells:
            if c["row"] != ridx or c["col"] not in day_cols:
                continue
            crop = img_main[c["y1"]:c["y2"], c["x1"]:c["x2"]]
            r_out, _ = run_ocr(crop)
            tokens += [t for t in texts_from_paddle(r_out) if t]
        if tokens:
            hits = sum(1 for t in tokens if is_price_like(t))
            letters = sum(1 for t in tokens if re.search(r"[가-힣A-Za-z]", t))
            row_is_price[ridx] = (hits >= 1) and (hits >= int(0.6*len(tokens))) and (letters <= 1)
        else:
            row_is_price[ridx] = False
    dbg("row_is_price ->", {r: row_is_price[r] for r in sorted(rows) if row_is_price[r]})

    # --- 출력: 슬롯별로 메뉴 먼저, 그 다음 가격 ---
    SLOT_ORDER = ["조식","코너A","코너B","석식"]

    for cidx in day_cols:
        day = col2day.get(cidx, "?")
        dbg(f"---- 열 시작: {day} (col={cidx}) ----")
        col_cnt = 0

        for slot in SLOT_ORDER:
            # 이 슬롯에 속한 행(위→아래 자연 순)
            slot_rows = [r for r in sorted(rows) if row2slot.get(r) == slot]

            # 1) 메뉴(가격 아님)
            for ridx in slot_rows:
                if row_is_price.get(ridx, False):
                    continue
                # (row, col) 셀 출력
                for cell in cells:
                    if cell["row"] != ridx or cell["col"] != cidx:
                        continue
                    crop = img_main[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
                    r_out, _ = run_ocr(crop)
                    lines = ocr_lines_from_result(r_out)
                    for line in lines:
                        text = normalize_text(line)
                        if not text or any(kw in text for kw in BLOCK_KWS):
                            continue
                        print(f"[{day}][{slot}] {text}")
                        any_printed = True
                        col_cnt += 1

            # 2) 가격
            for ridx in slot_rows:
                if not row_is_price.get(ridx, False):
                    continue
                for cell in cells:
                    if cell["row"] != ridx or cell["col"] != cidx:
                        continue
                    crop = img_main[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
                    r_out, _ = run_ocr(crop)
                    lines = ocr_lines_from_result(r_out)
                    for line in lines:
                        text = normalize_text(line)
                        if not text or any(kw in text for kw in BLOCK_KWS):
                            continue
                        print(f"[{day}][가격] {text}")  # 가격은 슬롯 라벨 대신 [가격]으로 명시
                        any_printed = True
                        col_cnt += 1

        dbg(f"---- 열 종료: {day} (출력 {col_cnt}건) ----")

    if not any_printed:
        print("[WRN] 출력할 텍스트가 없습니다.")

    dbg("apply_grid_and_debug -> cells", len(cells))
    return cells


def sanitize_csv_field(v):
    if v is None:
        return ""
    s = str(v)
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def sanitize_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            out[k] = sanitize_csv_field(v)
        else: out[k] = v
    return out


# ===== Selenium =====
options = Options()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)

def make_requests_session_from_driver(driver) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": driver.execute_script("return navigator.userAgent;"),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": driver.current_url
    })
    for c in driver.get_cookies():
        try:
            s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path","/"))
        except Exception:
            s.cookies.set(c["name"], c["value"])
    return s

def get_image() -> str:
    try:
        el = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".view_cont img, .view_cont .img img"))
        )
        return el.get_attribute("src") or ""
    except TimeoutException:
        pass
    imgs = driver.find_elements(By.CSS_SELECTOR, "img[src]")
    if imgs:
        def area(e):
            try:
                w = int(e.get_attribute("naturalWidth") or e.get_attribute("width") or 0)
                h = int(e.get_attribute("naturalHeight") or e.get_attribute("height") or 0)
                return w*h
            except Exception:
                return 0
        imgs.sort(key=area, reverse=True)
        src = imgs[0].get_attribute("src")
        if src: return src
    if "fileView" in driver.current_url:
        return driver.current_url
    for d in driver.find_elements(By.CSS_SELECTOR, '[style*="background-image"]'):
        style = d.get_attribute("style") or ""
        m = re.search(r'url\(["\']?(.*?)["\']?\)', style)
        if m: return urljoin(driver.current_url, m.group(1))
    raise NoSuchElementException("이미지 src를 찾지 못했습니다.")

def download_image_bytes(img_url: str, session: requests.Session) -> bytes:
    if not img_url: raise RuntimeError("img_url 비어 있음")
    if img_url.startswith("data:"):
        comma = img_url.find(",")
        if comma != -1:
            b64 = img_url[comma+1:]; return base64.b64decode(b64)
    url = urljoin(BASE, img_url)
    session.headers["Referer"] = driver.current_url
    r = session.get(url, timeout=(5,20))
    dbg(f"GET {url} | status={r.status_code} | bytes={len(r.content)} | ctype={r.headers.get('Content-Type','')}")
    r.raise_for_status()
    if not r.content: raise RuntimeError("이미지 응답 바이트 0")
    return r.content

# ===== 실행 =====
with open(OP_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id","날짜(요일)","운영시간대","구분","가격","식단"],
        lineterminator="\r\n"
    )
    writer.writeheader()

    page = 1
    while True:
        driver.get(LIST_URL.format(page))
        time.sleep(0.8)
        session = make_requests_session_from_driver(driver)

        found_any = False
        max_year = 0

        rows = driver.find_elements(
            By.XPATH,
            "//*[@id='contents']/div[3]/div[1]/table/tbody/tr[position()>=1 and position()<=10]"
        )
        if not rows: break

        for row in rows:
            try:
                title_a = row.find_element(By.CLASS_NAME, "td_tit").find_element(By.TAG_NAME, "a")
                href  = title_a.get_attribute("href") or ""
            except:
                continue

            try:
                date_td = row.find_element(
                    By.XPATH, ".//td[contains(@class,'td_write')]/following-sibling::td[1]"
                )
                date_str = date_td.text.strip()
            except:
                date_str = None

            if not date_str or not DATE_RE.fullmatch(date_str):
                continue

            year = int(date_str[:4])
            max_year = max(max_year, year)
            if year != 2025:
                continue
            found_any = True

            m = DETAIL_HREF_RE.search(href)
            if not m: continue
            post_id = m.group(1)
            detail_url = urljoin(BASE, href)

            driver.execute_script("window.open(arguments[0]);", detail_url)
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(0.5)

            try:
                session = make_requests_session_from_driver(driver)
                img = get_image()
                img_bytes = download_image_bytes(img, session)

                np_arr = np.frombuffer(img_bytes, np.uint8)
                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if np_img is None:
                    raise RuntimeError("이미지 디코드 실패")

                cells = apply_grid_and_debug(np_img, post_id=post_id, out_dir=OUT_DIR)

            finally:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])

        if not found_any and max_year and max_year < 2025:
            break
        page += 1

driver.quit()
