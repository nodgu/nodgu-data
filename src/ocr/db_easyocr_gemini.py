import easyocr
import requests
from io import BytesIO
from PIL import Image
# from correction import correct_text_with_et5t2t, correct_text_with_kogpt2, correct_text_with_kogrammar
# from test.gemini_api import gemini_prettier
# import psycopg
# from dotenv import load_dotenv
import os
import json
from requests.exceptions import JSONDecodeError

def ocr(url: str):
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    custom_dict = ['취업박람회', '쿠폰', '채용상담', '동국대학교', 'dongguk', 'university', '교수학습혁신센터', '2차면접', '싱잉볼', '인증', '캡처', '콘텐츠', '그래픽', '군악', '불교청년지도자육성', '서울캠퍼스', '선발', '많은', '바랍니다.', '오리엔테이션', '면접', '시간', '스트레스', '프로그래밍', '비대면', '숲에서', '석박사통합과정', '첨단분야', '글로벌', '화공생물공학과', '미래융합', 'AI융합', '열린전공', '쿨하게', '이슈', '알라딘', '있습니다', '바꾸는', '업사이클링', '학습역량', '신호정보/전자전운용병', '\'25-2', '캡스톤디자인', '2개', '교과목', '참여', 'AI활용', '공익', '콘텐츠', '전형', '이었던', 'nDRIMS', '하오니', '증명서', '졸엽예정', '대출', '증빙', '가구원', '전원', '완료', '온라인', '취약', '소득분위', '기초생활수급자', '차상위계층', '대출장학금', '학자금', '신청', '종료', '공모전', '멘토링', '인도네시아', '자카르타', '현지인', '자치도', '있', '드릴', '있으니', '됩니댜', '다르마칼리지', '']

    # url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1724718253381.png&contentType=image"
    response = requests.get(url) # 이미지 url
    img = Image.open(BytesIO(response.content))

    img_byte = BytesIO() # 빈 이미지 생성
    img.convert('RGB').save(img_byte, format='JPEG') # 이미지 전처리해서 빈 이미지에 저장

    result = reader.readtext(img_byte.getvalue()) # 배열로 출력


    # for (bbox, text) in result:
    #     print(text)

    texts = [text for (_, text, _) in result ]
    text = '\n'.join(texts)
    print(text)

    return

    # print("==================================")

    # return gemini_prettier(text)



# def get_data():
#     load_dotenv()

#     db = psycopg.connect(
#         host=os.environ["POSTGRES_HOST"],
#         dbname=os.environ["POSTGRES_DB"],
#         user=os.environ["POSTGRES_USER"],
#         password=os.environ["POSTGRES_PASSWORD"],
#         port=os.environ["POSTGRES_PORT"]
#     )

#     cursor = db.cursor()
#     cursor.execute("SELECT id, imgs, tdindex FROM notice_notice WHERE jsonb_array_length(ocr_data) = 0 AND jsonb_array_length(imgs) > 0 LIMIT 3;")

#     for (id, imgs, tdindex) in cursor.fetchall():
#         print(id, imgs)

#         ocr_data = []
#         for img in imgs:
#             print(img)
#             ocr_data.append(ocr(img))
#         new_tdindex = tdindex + "\n" + "\n".join(ocr_data)
#         cursor.execute("UPDATE notice_notice SET ocr_data = %s, tdindex = %s WHERE id = %s;", (json.dumps(ocr_data), new_tdindex, id))
#         db.commit()
#     cursor.close()


# if __name__ == "__main__":
#     get_data()


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
    # 200번대가 아닐 때
    print(f"HTTP 오류: {e}")
except JSONDecodeError as e:
    # 응답 본문이 JSON 형식이 아닐 때
    print(f"JSON 디코딩 오류: {e}. 응답 텍스트: '{response.text}'")
except requests.exceptions.RequestException as e:
    # 연결 문제와 같은 기타 requests
    print(f"요청 오류: {e}")

for item in data:
    imgs = item['imgs']
    # print("새 이미지 리스트")
    for image in imgs:
        # print(f'새로운 이미지 url: {image}')
        ocr(image)