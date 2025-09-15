from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
import time
import csv
import os
from urllib.parse import urljoin

BASE = "https://dorm.dongguk.edu"
LIST_URL = BASE + "/article/food/list?pageIndex={}"
DETAIL_URL = BASE + "/article/food/detail/{}"
DETAIL_HREF_RE = re.compile(r"/article/food/detail/(\d+)")

# === 출력 경로 ===
OUT_DIR = "output2"
os.makedirs(OUT_DIR, exist_ok=True)
OP_CSV = os.path.join(OUT_DIR, "운영시간.csv")

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# 정규식
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
POST_ID_RE = re.compile(r"goDetail\((\d+)\)")
TIME_RE = re.compile(r"(\d{1,2}:\d{2}\s*~\s*\d{1,2}:\d{2}|X|CLOSED)", re.IGNORECASE)

# === 유틸: 사진 가져오기 ===
def get_image():
    img = driver.find_element(By.CLASS_NAME, "view_cont").find_element(By.TAG_NAME, "img")
    src = img.get_attribute("src")
    return src

# === 메인: 목록 순회 ===
op_candidates = []          # (post_id, title, date_str)

page = 1
while True:
    driver.get(LIST_URL.format(page))
    time.sleep(1)

    found_2025 = False
    max_year = 0

    rows = []
    for i in range(2, 12):
        try:
            row = driver.find_element(By.XPATH, f'//*[@id="contents"]/div[3]/div[1]/table/tbody/tr[{i}]')
            rows.append(row)
        except:
            continue

    if not rows:
        break

    for row in rows:
        try:
            # 제목 a
            title_a = row.find_element(By.CLASS_NAME, "td_tit").find_element(By.TAG_NAME, "a")
            title = title_a.text.strip()
        except:
            continue

        # 날짜 추출
        tds = row.find_elements(By.TAG_NAME, "td")
        date_str = None
        for td in tds:
            txt = td.text.strip()
            if DATE_RE.fullmatch(txt):
                date_str = txt
                break
        if not date_str:
            continue

        year = int(date_str[:4])
        max_year = max(max_year, year)
        if year != 2025:
            continue
        found_2025 = True

        # href에서 바로 추출
        href = title_a.get_attribute("href") or ""
        m = DETAIL_HREF_RE.search(href)
        if not m:
            continue
        post_id = m.group(1)
        detail_url = urljoin(BASE, href)

        # 제목 분기
        if "운영시간" in title:
            # 최신 1건 선별을 위해 후보만 쌓아둠
            op_candidates.append((post_id, title, date_str))

        elif "주간 식단표" in title:
            # 새 탭 열기
            driver.execute_script("window.open(arguments[0]);", detail_url)
            driver.switch_to.window(driver.window_handles[-1])  # 새 탭으로 전환
            time.sleep(0.4)

            # 탭 닫기
            driver.close()
            driver.switch_to.window(driver.window_handles[0]) 

    if not found_2025 and max_year and max_year < 2025:
        break
    page += 1

# === 운영시간: 최신 1건만 처리 ===
if op_candidates:
    # 날짜 기준 내림차순 정렬 후 첫 번째(최신)
    op_candidates.sort(key=lambda x: x[2].replace("-", ""), reverse=True)
    latest_post_id, latest_title, latest_date = op_candidates[0]

    # 디테일 페이지 이동
    latest_detail = DETAIL_URL.format(latest_post_id)
    driver.get(latest_detail)
    time.sleep(0.8)

    # 본문 이미지 가져와 OCR
    img = get_image()
    if img:
        # OCR
        records = []

        # CSV 저장(덮어쓰기)
        with open(OP_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["구분","세부","평일 운영시간","주말 및 공휴일"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[운영시간] 최신 글({latest_date}, {latest_title}) → CSV 저장 완료: {OP_CSV}")
    else:
        print("[운영시간] 본문 이미지를 찾지 못했습니다:", latest_detail)

else:
    print("[운영시간] 2025년 글(운영시간) 후보 없음")

driver.quit()
