from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import requests
import os
from urllib.parse import urljoin
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

last_week_date = 20090830
global current_week_date

base_url = "https://dgucoop.dongguk.edu"
url = base_url + "/store/store.php?w=4&l=2&j={}"

# restaurants = ["SANGNOK1", "SANGNOK2", "SANGNOK3", "koyang", "DFLEX", "NAMSAN"]
restaurant_id = {
    "SANGNOK1": 1,
    "SANGNOK2": 2,
    "SANGNOK3": 3,
    "koyang": 4,
    "DFLEX": 5,
    "NAMSAN": 6,
}

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)


restaurants = []

def to_menu(id, campus, name, address):
    menu = {
        "restaurant_id": id,
        "univ": "Dongguk",
        "campus": campus,
        "name": name,
        "address": address,
    }
    restaurants.append(menu)
    return
j = 0
while True:
    restaurants  = []
    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
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

        el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(2) > td')
        sangnok3_name = el.get_text(strip=True) if el and el.get_text(strip=True) else None
        to_menu(3, "seoul", sangnok3_name, "상록원 3층 위치" if sangnok3_name else None)

        el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(7) > td')
        sangnok2_name = el.get_text(strip=True) if el and el.get_text(strip=True) else None
        to_menu(2, "seoul", sangnok2_name, "상록원 2층 위치" if sangnok2_name else None)

        el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(16) > td')
        sangnok1_name = el.get_text(strip=True) if el and el.get_text(strip=True) else None
        to_menu(1, "seoul", sangnok1_name, "상록원 1층 위치" if sangnok1_name else None)

        el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(24) > td')
        koyang_name = el.get_text(strip=True) if el and el.get_text(strip=True) else None
        to_menu(4, "ilsan", koyang_name, "고양학사 위치" if koyang_name else None)

        try:
            response = requests.post(f"{API_HOST}/api/v1/menu/menus", json=restaurants, timeout=10)
            if response.status_code == 200:
                print(f"Save to DB")
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