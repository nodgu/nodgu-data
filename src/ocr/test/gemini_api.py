import google.generativeai as genai
import os
from dotenv import load_dotenv


# prompt = input("프롬프트 입력하세요: ")

text1 = """
Creator Crew of Dongguk
크크동 모집
동국대학교 공식 유튜브 크리에이터률 모집합니다
01 지원자격
영상 출연 촬영(소니 미러리스 카메라활용) 편집
(ADOBEPREMIER PRO 프로그램활용) 중1가지 이상
울할 수있는 학생
2개 학기여름 겨울방학 포함) 동안 꾸준한 활동이 가능
한재학생 (휴학생 지원 불가)
정기회의 참석이 가능한 학생(격주 월요일 18시 30분)
직전학기 취특학점 12학점 이상이수 덧 평점평균 2.001
상 취특한 학생
02 우대사항
미러리스카메라와 ADOBE PREMIER PRO틀 함께 다물 수앞는 학생
동국대학교 홍보에 대한아이디어가 풍부한 학생
03 활동혜택
굳렌- 제작 건당 장학금 지급 (통9소픔 차등있음)
촬영 장비 및 편집 프로그램 지원
활동완료 후 수로증 지급
04모집절차
1차: 서류전형(자기소개서 제출)
~2024.09.03.
*철부되 지원서 작성 후E-MAIL 접수
(leeray@donggukedu)
2차: 면접 전형대면 진행)
2024.09.06.
오리언테이선 일정
2024.09.09.18.30
"""

text1 = """
E N V|R0 N M E N T
S 0 C l AL
G OV E R N A N C E
중구 X 동 국 대
'우리도네
ESG
서울시 중구의 주요 산업

인쇄, 봉제산업
공정 과정예서 버려지논 자원이 새로 태어낫어요
자투리| 종이가 메모패드와 다이어리로,
자투리 천이 인형과 인형옷으로!
' ?
O J
동국대 'ESG X 돌봄 프로적트' 기사 구경하기!
'업사이클림
"""

def gemini_prettier(text):
    prompt = f"{text} 이거 OCR에서 얻은 글인데 잘못된 글자도 수정하고 보기 좋게 만들어봐"

    load_dotenv()

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="너는 OCR에서 발생한 오류를 수정하는 거야. 이제 부터 내가 주는 모든 글을 읽고 오류를 수정하고 보기 좋게 정리해줘. 정리된 내용만 보여주고 별도의 설명같은 것은 필요없어")
    response = model.generate_content(text)
    return response.text

def gemini_input():
    prompt = input("프롬프트: ")

    load_dotenv()

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print(gemini_input())