from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import requests
import os
from bs4 import BeautifulSoup
from pprint import pprint
from datetime import date, datetime
from dotenv import load_dotenv

# .env 파일 load
load_dotenv()
API_HOST = os.getenv("API_HOST")

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def to_int(week):
    week_date = re.search(r'(\d{4})\.(\d{2})\.(\d{2})', week)
    week_date_int = ''.join(week_date.groups())
    return int(week_date_int)

def mapping(name: str):
    if not name: return None, None
    raw = name

    if "상록원" in raw:
        if "3" in raw: return 3, "seoul"
        if "2" in raw: return 2, "seoul"
        if "1" in raw: return 1, "seoul"
    if ("누리터" in raw) or ("일산" in raw) or ("BMC" in raw): return 4, "ilsan"
    if "FLEX" in raw: return 5, "seoul"
    if "남산" in raw: return 6, "seoul"
    if "가든쿡" in raw: return 7, "seoul"
    if "그루터기" in raw: return 8, "seoul"
    if "아리수" in raw: return 9, "seoul"
    if "코끼리" in raw: return 10, "seoul"
    if "그루터기" in raw: return 8, "seoul"
    return None, None

def get_or_none(seq, idx):
    return seq[idx] if idx < len(seq) and seq[idx] else None

def get_existing_ids():
    try:
        r = requests.get(f"{API_HOST}/api/v1/menu/restaurants/ids", timeout=10)
        r.raise_for_status()
        data = r.json()
        # 숫자 보장
        return {int(x) for x in data if x is not None}
    except Exception as e:
        print(f"기존 restaurantId 조회 실패: {e}")
        return set()

seen_in_batch = set()

last_week_date = 20090830
global current_week_date

base_url = "https://dgucoop.dongguk.edu"
url = base_url + "/store/store.php?w=4&l=2&j={}"

restaurants = []

def to_menu(id, campus, name):
    if not id or not name: 
        return
    if id in seen_in_batch:
        return
    restaurants.append({
        "restaurant_id": id,
        "univ": "Dongguk",
        "campus": campus,
        "name": name,
    })
    seen_in_batch.add(id)
    return

j = 0
existing_ids = get_existing_ids() 

while True:
    restaurants  = []
    response = requests.get(url.format(j))

    if response.status_code == 200:
        html = response.content.decode("cp949")
        soup = BeautifulSoup(html, "html.parser")
        
        cw = soup.select_one(".menu_date")
        current_week = cw.get_text(strip=True) if cw else ""
        current_week_date = to_int(current_week)
        if current_week_date is None:
            print("주차 파싱 실패. 중단합니다.")
            break
        if current_week_date < last_week_date:
            print("break 조건 달성. 중단합니다.")
            break
        
        table = soup.select_one('#sdetail > table:nth-of-type(2)')
        names = [td.get_text(strip=True) for td in (table.select('td.menu_st'))]
        for name in names:
            id, campus = mapping(name)
            if id is None:
                continue
            if id in existing_ids:
                continue
            to_menu(id, campus, name)

        if restaurants:
            try:
                response = requests.post(f"{API_HOST}/api/v1/menu/restaurants", json=restaurants, timeout=10)
                if response.status_code == 201:
                    print(f"Save to DB, {restaurants}")
                else:
                    print(f"API 에러 - 상태코드: {response.status_code}, 응답: {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"연결 에러: Spring 서버가 실행 중인지 확인하세요 ({API_HOST})")
            except requests.exceptions.Timeout:
                print(f"타임아웃 에러")
            except Exception as e:
                print(f"예상치 못한 에러: {e}")
    else:
        pprint(response)

    j -= 1

last_week_date = current_week_date