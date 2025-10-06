from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
import unicodedata
import requests
import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pprint import pprint
from datetime import timedelta, datetime, date
from dotenv import load_dotenv

# .env 파일 load
load_dotenv()
API_HOST = os.getenv("API_HOST")

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def nth_text_of(el, n):
    if el is None:
        return None
    texts = [t for t in el.find_all(string=True, recursive=False)]
    return texts[n-1].strip() if len(texts) >= n else None

def to_int(week):
    week_date = re.search(r'(\d{4})\.(\d{2})\.(\d{2})', week)
    week_date_int = ''.join(week_date.groups())
    return int(week_date_int)

def parse_week_start(text: str) -> datetime:
    m = re.search(r'(\d{4}\.\d{2}\.\d{2})', text)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y.%m.%d")

def find_food(base_css, start_idx):
    foods = []
    j = 1
    el = soup.select_one(base_css)  # base_css는 span:nth-of-type(1)까지를 가리킴
    if el is None:
        return foods, start_idx
    
    # start_idx만큼 오른쪽 형제 span으로 이동
    while el is not None and j < start_idx:
        el = el.find_next_sibling('span')
        j += 1
    if el is None:
        return foods, j
    
    # 연속된 span들을 순회하며 메뉴 수집
    while el is not None:
        s = el.get_text(strip=True)
        food = re.split(r'\s|\(|\d', unicodedata.normalize('NFKC', s).replace('\u00A0',' '), 1)[0].rstrip()
        if food and (food[0] in ('(', '*', '-') or '0' <= food[0] <= '9'):
            el = el.find_next_sibling('span'); j += 1
            continue
        foods.append(food)
        el = el.find_next_sibling('span'); j += 1

    return foods, j


base_url = "https://dgucoop.dongguk.edu"
url = base_url + "/store/store.php?w=4&l=2&j={}"

restaurants = ["SANGNOK1", "SANGNOK2", "SANGNOK3", "koyang", "DFLEX", "NAMSAN"]
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

idx = 0
driver.get(url.format(idx))
current_week = driver.find_element(By.CLASS_NAME, "menu_date").text.strip()

global current_week_date
current_week_date = to_int(current_week)
last_week_date = 20090830

driver.quit()

foods = []
def to_food(restaurant_id, date, time, corner, activated, food):
    menu = {
        "restaurant_id": restaurant_id,
        "date": date,
        "time": time,
        "corner": corner,
        "activated": activated,
        "food": food,
    }
    foods.append(menu)

while True:
    print(f"{idx}")
    print(f"Fetching Url: {url}")
    response = requests.get(url.format(idx))
    foods = []

    if response.status_code == 200:
        html = response.text
        global soup
        soup = BeautifulSoup(html, "html.parser")

        cw = soup.select_one(".menu_date")
        current_week = cw.get_text(strip=True) if cw else ""
        current_week_date = to_int(current_week)
        week_start = parse_week_start(current_week)

        if (current_week_date < last_week_date):
            print("break 조건 달성. 중단합니다.")
            break
        
        base_css = ""
        
        for i in range(2, 9):
            today = (week_start + timedelta(days=i-2)).strftime("%Y-%m-%d")
            print(today)
            
            print('---------- SANGNOK3 ----------')
            sangnok3_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(2) > td')
            sangnok3_corner1_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(3) > td:nth-of-type(1)')

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(3) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok3_corner1_lunch, _ = find_food(base_css, 1)

            sangnok3_corner1_lunch_time_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(3) > td:nth-of-type(2)')
            to_food(3, today,
                    sangnok3_corner1_lunch_time_el.get_text(strip=True) if sangnok3_corner1_lunch_time_el else None,
                    sangnok3_corner1_name_el.get_text(strip=True) if sangnok3_corner1_name_el else None,
                    None,
                    sangnok3_corner1_lunch)

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(4) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok3_corner1_dinner, _ = find_food(base_css, 1)
            sangnok3_corner1_dinner_time_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(4) > td:nth-of-type(1)')
            to_food(3, today,
                    sangnok3_corner1_dinner_time_el.get_text(strip=True) if sangnok3_corner1_dinner_time_el else None,
                    sangnok3_corner1_name_el.get_text(strip=True) if sangnok3_corner1_name_el else None,
                    None,
                    sangnok3_corner1_dinner)
            
            sangnok3_corner2_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(5) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(5) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok3_corner2_lunch, _ = find_food(base_css, 1)
            sangnok3_corner2_lunch_time_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(5) > td:nth-of-type(2)')

            sangnok3_corner2_lunch_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(5) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok3_corner2_lunch_act_txt = nth_text_of(sangnok3_corner2_lunch_act_el, _)
            sangnok3_corner2_lunch_act = f"중식: {sangnok3_corner2_lunch_act_txt[2:]}" if sangnok3_corner2_lunch_act_txt else None

            to_food(3, today,
                    sangnok3_corner2_lunch_time_el.get_text(strip=True) if sangnok3_corner2_lunch_time_el else None,
                    sangnok3_corner2_name_el.get_text(strip=True) if sangnok3_corner2_name_el else None,
                    sangnok3_corner2_lunch_act,
                    sangnok3_corner2_lunch)

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(6) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok3_corner2_dinner, _ = find_food(base_css, 1)
            sangnok3_corner2_dinner_time_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(6) > td:nth-of-type(1)')

            sangnok3_corner2_dinner_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(6) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok3_corner2_dinner_act_txt = nth_text_of(sangnok3_corner2_dinner_act_el, _)
            sangnok3_corner2_dinner_act = f"석식: {sangnok3_corner2_dinner_act_txt[2:]}" if sangnok3_corner2_dinner_act_txt else None

            to_food(3, today,
                    sangnok3_corner2_dinner_time_el.get_text(strip=True) if sangnok3_corner2_dinner_time_el else None,
                    sangnok3_corner2_name_el.get_text(strip=True) if sangnok3_corner2_name_el else None,
                    sangnok3_corner2_dinner_act,
                    sangnok3_corner2_dinner)
            
            print('---------- SANGNOK2 ----------')
            sangnok2_corner1_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(8) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(8) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok2_corner1_lunch, _ = find_food(base_css, 1)

            sangnok2_corner1_lunch_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(8) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok2_corner1_lunch_act = nth_text_of(sangnok2_corner1_lunch_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner1_name_el.get_text(strip=True) if sangnok2_corner1_name_el else None,
                    sangnok2_corner1_lunch_act,
                    sangnok2_corner1_lunch)

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(9) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok2_corner1_dinner, _ = find_food(base_css, 1)

            sangnok2_corner1_dinner_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(9) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok2_corner1_dinner_act = nth_text_of(sangnok2_corner1_dinner_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner1_name_el.get_text(strip=True) if sangnok2_corner1_name_el else None,
                    sangnok2_corner1_dinner_act,
                    sangnok2_corner1_dinner)
            
            sangnok2_corner2_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(10) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(10) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok2_corner2_lunch, _ = find_food(base_css, 1)

            sangnok2_corner2_lunch_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(10) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok2_corner2_lunch_act = nth_text_of(sangnok2_corner2_lunch_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner2_name_el.get_text(strip=True) if sangnok2_corner2_name_el else None,
                    sangnok2_corner2_lunch_act,
                    sangnok2_corner2_lunch)
            
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(11) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok2_corner2_dinner, _ = find_food(base_css, 1)

            sangnok2_corner2_dinner_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(11) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok2_corner2_dinner_act = nth_text_of(sangnok2_corner2_dinner_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner2_name_el.get_text(strip=True) if sangnok2_corner2_name_el else None,
                    sangnok2_corner2_dinner_act,
                    sangnok2_corner2_dinner)

            sangnok2_corner3_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(12) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(12) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok2_corner3_lunch, _ = find_food(base_css, 1)

            sangnok2_corner3_lunch_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(12) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok2_corner3_lunch_act = nth_text_of(sangnok2_corner3_lunch_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner3_name_el.get_text(strip=True) if sangnok2_corner3_name_el else None,
                    sangnok2_corner3_lunch_act,
                    sangnok2_corner3_lunch)
            
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(13) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok2_corner3_dinner, _ = find_food(base_css, 1)

            sangnok2_corner3_dinner_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(13) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok2_corner3_dinner_act = nth_text_of(sangnok2_corner3_dinner_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner3_name_el.get_text(strip=True) if sangnok2_corner3_name_el else None,
                    sangnok2_corner3_dinner_act,
                    sangnok2_corner3_dinner)

            sangnok2_corner4_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(14) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(14) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok2_corner4_lunch, _ = find_food(base_css, 1)

            sangnok2_corner4_lunch_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(14) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok2_corner4_lunch_act = nth_text_of(sangnok2_corner4_lunch_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner4_name_el.get_text(strip=True) if sangnok2_corner4_name_el else None,
                    sangnok2_corner4_lunch_act,
                    sangnok2_corner4_lunch)
            
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(15) > td:nth-of-type({i}) > span:nth-of-type(1)'
            sangnok2_corner4_dinner, _ = find_food(base_css, 1)

            sangnok2_corner4_dinner_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(15) > td:nth-of-type({i}) > span:nth-of-type(1)'
            )
            sangnok2_corner4_dinner_act = nth_text_of(sangnok2_corner4_dinner_act_el, 1)

            to_food(2, today, "중석식",
                    sangnok2_corner4_name_el.get_text(strip=True) if sangnok2_corner4_name_el else None,
                    sangnok2_corner4_dinner_act,
                    sangnok2_corner4_dinner)

            print('---------- SANGNOK1 ----------')
            sangnok1_corner1_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(17) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(17) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner1, _ = find_food(base_css, 1)

            sangnok1_corner1_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(17) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner1_act = nth_text_of(sangnok1_corner1_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner1_name_el.get_text(strip=True) if sangnok1_corner1_name_el else None,
                    sangnok1_corner1_act,
                    sangnok1_corner1)

            sangnok1_corner2_name_cell_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(18) > td:nth-of-type(2) > span:nth-of-type(1)')
            sangnok1_corner2_name_txt = nth_text_of(sangnok1_corner2_name_cell_el, 1)
            sangnok1_corner2_name = sangnok1_corner2_name_txt[4:-5] if sangnok1_corner2_name_txt else None

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(18) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner2, _ = find_food(base_css, 1)

            sangnok1_corner2_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(18) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner2_act = nth_text_of(sangnok1_corner2_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner2_name,
                    sangnok1_corner2_act,
                    sangnok1_corner2)

            sangnok1_corner3_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(19) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(19) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner3, _ = find_food(base_css, 1)

            sangnok1_corner3_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(19) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner3_act = nth_text_of(sangnok1_corner3_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner3_name_el.get_text(strip=True) if sangnok1_corner3_name_el else None,
                    sangnok1_corner3_act,
                    sangnok1_corner3)

            sangnok1_corner4_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(20) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(20) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner4, _ = find_food(base_css, 1)

            sangnok1_corner4_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(20) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner4_act = nth_text_of(sangnok1_corner4_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner4_name_el.get_text(strip=True) if sangnok1_corner4_name_el else None,
                    sangnok1_corner4_act,
                    sangnok1_corner4)

            sangnok1_corner5_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(21) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(21) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner5, _ = find_food(base_css, 1)

            sangnok1_corner5_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(21) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner5_act = nth_text_of(sangnok1_corner5_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner5_name_el.get_text(strip=True) if sangnok1_corner5_name_el else None,
                    sangnok1_corner5_act,
                    sangnok1_corner5)

            sangnok1_corner6_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(22) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(22) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner6, _ = find_food(base_css, 1)

            sangnok1_corner6_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(22) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner6_act = nth_text_of(sangnok1_corner6_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner6_name_el.get_text(strip=True) if sangnok1_corner6_name_el else None,
                    sangnok1_corner6_act,
                    sangnok1_corner6)

            sangnok1_corner7_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(23) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(23) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            sangnok1_corner7, _ = find_food(base_css, 1)

            sangnok1_corner7_act_el = soup.select_one(
                f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(23) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            )
            sangnok1_corner7_act = nth_text_of(sangnok1_corner7_act_el, 1)

            to_food(1, today, "중석식",
                    sangnok1_corner7_name_el.get_text(strip=True) if sangnok1_corner7_name_el else None,
                    sangnok1_corner7_act,
                    sangnok1_corner7)

            print('---------- KOYANG ----------')
            koyang_corner1_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(25) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(25) > td:nth-of-type({i}) > span:nth-of-type(1)'
            koyang_corner1, _ = find_food(base_css, 1)
            to_food(4, today, None,
                    koyang_corner1_name_el.get_text(strip=True) if koyang_corner1_name_el else None,
                    None, koyang_corner1)
            
            koyang_corner2_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(26) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(26) > td:nth-of-type({i+1}) > span:nth-of-type(1)'
            koyang_corner2_lunch, _ = find_food(base_css, 1)
            to_food(4, today, None,
                    koyang_corner2_name_el.get_text(strip=True) if koyang_corner2_name_el else None,
                    None, koyang_corner2_lunch)

            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(27) > td:nth-of-type({i}) > span:nth-of-type(1)'
            koyang_corner2_dinner, _ = find_food(base_css, 1)
            to_food(4, today, None,
                    koyang_corner2_name_el.get_text(strip=True) if koyang_corner2_name_el else None,
                    None, koyang_corner2_dinner)
            
            koyang_corner3_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(28) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(28) > td:nth-of-type({i}) > span:nth-of-type(1)'
            koyang_corner3, _ = find_food(base_css, 1)
            to_food(4, today, None,
                    koyang_corner3_name_el.get_text(strip=True) if koyang_corner3_name_el else None,
                    None, koyang_corner3)

            koyang_corner4_name_el = soup.select_one('#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(29) > td:nth-of-type(1)')
            base_css = f'#sdetail > table:nth-of-type(2) > tbody > tr:nth-of-type(29) > td:nth-of-type({i}) > span:nth-of-type(1)'
            koyang_corner4, _ = find_food(base_css, 1)
            to_food(4, today, None,
                    koyang_corner4_name_el.get_text(strip=True) if koyang_corner4_name_el else None,
                    None, koyang_corner4)

            try:
                response = requests.post(f"{API_HOST}/api/v1/menu/menus", json=foods, timeout=10)
                if response.status_code == 200:
                    print("Save to DB")
                else:
                    print(f"API 에러 - 상태코드: {response.status_code}, 응답: {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"연결 에러: Spring 서버가 실행 중인지 확인하세요 ({API_HOST})")
            except requests.exceptions.Timeout:
                print("타임아웃 에러")
            except Exception as e:
                print(f"예상치 못한 에러: {e}")

    else:
        pprint(response)

    idx -= 1
