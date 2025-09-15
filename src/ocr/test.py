import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import easyocr
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

def preprocess_image(img):
    # PIL을 OpenCV 형태로 변환
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 이미지 크기 확대 (해상도 향상)
    height, width = img_cv.shape[:2]
    img_cv = cv2.resize(img_cv, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    denoised = cv2.medianBlur(gray, 3)
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 텍스트 개선
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(processed)

def improved_ocr(url: str):
    
    # 더 포괄적인 커스텀 사전
    custom_dict = [
        '취업박람회', '채용박람회', '쿠폰', '채용상담', '동국대학교', 
        'dongguk university', '서울캠퍼스', '경주캠퍼스',
        '학생처', '취업지원팀', '커리어개발센터', '상담', '면접',
        '이력서', '자기소개서', '인사담당자', '기업', '회사',
        '일시', '장소', '참가', '신청', '문의', '연락처'
    ]
    
    # GPU 사용 (가능한 경우)
    try:
        reader = easyocr.Reader(['ko', 'en'], gpu=True)
    except:
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
    
    # 이미지 다운로드
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # 이미지 전처리
    processed_img = preprocess_image(img)
    
    # OCR 실행 (여러 설정으로)
    img_byte = BytesIO()
    processed_img.convert('RGB').save(img_byte, format='JPEG', quality=95)
    
    # 기본 OCR
    result1 = reader.readtext(
        img_byte.getvalue(),
        detail=0,  # 좌표 정보 제외
        paragraph=True,  # 문단 단위로 인식
        width_ths=0.7,  # 텍스트 폭 임계값
        height_ths=0.7  # 텍스트 높이 임계값
    )
    
    # 더 엄격한 설정으로 한 번 더
    result2 = reader.readtext(
        img_byte.getvalue(),
        detail=0,
        paragraph=False,
        width_ths=0.9,
        height_ths=0.9
    )
    
    # 결과 병합 및 중복 제거
    all_texts = result1 + result2
    unique_texts = list(dict.fromkeys(all_texts))  # 순서 유지하며 중복 제거
    
    return '\n'.join(unique_texts)


def ocr(url: str):
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    custom_dict = ['취업박람회', '쿠폰', '채용상담', '동국대학교', 'dongguk university']

    # url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1724718253381.png&contentType=image"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    img_byte = BytesIO()
    img.convert('RGB').save(img_byte, format='JPEG')

    result = reader.readtext(img_byte.getvalue())


    # for (bbox, text) in result:
    #     print(text)

    texts = [text for (_, text, _) in result ]
    text = '\n'.join(texts)

    return text

def gemini_ocr(url: str):
    load_dotenv()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    model = genai.GenerativeModel("gemini-2.5-flash-lite", system_instruction="너는 OCR을 하는 거야. 이제 부터 내가 주는 모든 이미지를 읽고 보기 좋게 정리해줘. 정리된 내용만 보여주고 별도의 설명같은 것은 필요없어")
    response = model.generate_content(img)
    return response.text

url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1754368219403.png&contentType=image"

# print(ocr(url))
# print("-----------------------------------------")
# print(improved_ocr(url))
# print("-----------------------------------------")
print(gemini_ocr(url))