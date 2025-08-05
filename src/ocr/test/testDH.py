import requests
import json
from requests.exceptions import JSONDecodeError

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
    for image in imgs: