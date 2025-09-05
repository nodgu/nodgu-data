# paddleOCR.py

import re
import requests
from io import BytesIO
from PIL import Image
from requests.exceptions import JSONDecodeError
import numpy as np
import cv2
from paddleocr import PaddleOCR

# ----------------------------------------------------
# PaddleOCR 엔진 (지연 초기화: 첫 호출 시 생성)
# - lang='korean' : 한국어 인식 모델
# - use_angle_cls=True : 글자 각도 보정
# - res=True, rec=True 전체 파이프라인
# - use_gpu=False : EasyOCR 코드와 동일하게 CPU로 설정
# ----------------------------------
_OCR_ENGINE = None
def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = PaddleOCR(
            lang='korean',
            device='cpu',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _OCR_ENGINE

edit_re = re.compile(r'\.(?:jpg|jpeg|png|bmp|pdf)(?![a-zA-Z])', re.IGNORECASE)

def trim_url(url: str) -> str:
    m = None
    # URL 전체에서 마지막 확장자 매치 위치를 찾기 위해 finditer 사용
    for _m in edit_re.finditer(url):
        m = _m
    if m:
        return url[:m.end()]
    return url

def _extract_texts_v3x(predict_result) -> list[str]:
    """3.x predict() 출력에서 텍스트만 안전하게 뽑기"""
    texts = []
    if not predict_result:
        return texts
    for item in predict_result:
        data = getattr(item, "res", None)
        if data is None and isinstance(item, dict):
            data = item
        if not isinstance(data, dict):
            continue
        if "rec_texts" in data and isinstance(data["rec_texts"], list):
            texts.extend([str(t) for t in data["rec_texts"]])
        elif "rec_text" in data:  # 단일 인식 결과 형태
            texts.append(str(data["rec_text"]))
    return texts

def ocr(url: str):
    ocr_engine = _get_ocr_engine()
    # ocr() 맨 처음 (ocr_engine = _get_ocr_engine() 바로 다음)
    print(f"[DBG] start ocr | url={url}")
    trimmed = trim_url(url)
    print(f"[DBG] trimmed={trimmed}" + (" (changed)" if trimmed != url else " (same)"))

    # 이미지 가져오기
    response = requests.get(trimmed) # 이미지 url
    print(f"[DBG] GET done | status={response.status_code} | bytes={len(response.content)} | "
        f"content-type={response.headers.get('Content-Type','')} | final_url={response.url}")

    # 이미지 확인(다운로드 성공/용량 확인용)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(trimmed, headers=headers, timeout=(5, 20))
        print(f"[DBG] GET | status={r.status_code} | bytes={len(r.content)} | "
              f"content-type={r.headers.get('Content-Type','')} | final_url={r.url}")
        r.raise_for_status()
        if not r.content:
            print("[ERR] 응답 바이트가 0입니다.")
            return ""
        # 열기만 해서 포맷 로그 남김 (예: RGB/크기)
        img = Image.open(BytesIO(r.content))
        print(f"[DBG] PIL open | mode={img.mode} | size={img.size}")
    except Exception as e:
        print(f"[ERR] 이미지 확인 실패: {e}")
        return ""

    # 3.x 공식 문서대로 predict() 사용 (입력: URL 또는 ndarray 모두 가능)
    try:
        print("[DBG] call ocr_engine.predict(trimmed_url)")
        result = ocr_engine.predict(trimmed)
    except Exception as e:
        print(f"[ERR] PaddleOCR predict 실패: {e}")
        return "실패"

    texts = _extract_texts_v3x(result)
    print(f"[DBG] final lines={len(texts)}")
    if texts:
        print("\n".join(texts))
    else:
        print("[WRN] 텍스트가 검출되지 않았습니다.")
    return "\n".join(texts)


data = []
response = requests.get("https://api.nodgu.shop/api/v1/notice/notice/noOcrData")

try:
    response.raise_for_status()

    if response.text:
        responseJSON = response.json()
        data = responseJSON['data']
    else:
        print("요청 성공. 응답 본문 비어있음")

except requests.exceptions.HTTPError as e:
    print(f"HTTP 오류: {e}")
except JSONDecodeError as e:
    print(f"JSON 디코딩 오류: {e}. 응답 텍스트: '{response.text}'")
except requests.exceptions.RequestException as e:
    print(f"요청 오류: {e}")

for item in data:
    imgs = item['imgs']
    print("공지")
    for image in imgs:
        print("이미지:", image)
        ocr(image)
