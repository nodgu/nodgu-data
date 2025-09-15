import os, csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re, time
from urllib.parse import urljoin

BASE = "https://bmcdorm.dongguk.edu"
LIST_URL = BASE + "/article/menu/list?pageIndex={}"
DETAIL_HREF_RE = re.compile(r"/article/menu/detail/(\d+)")   # href에서 ID 추출

OUT_DIR = "output_BMC"
os.makedirs(OUT_DIR, exist_ok=True)
OP_CSV = os.path.join(OUT_DIR, "BMC.csv")

# 날짜 형식
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Selenium 설정
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

def get_image():
    img = driver.find_element(By.CLASS_NAME, "view_cont").find_element(By.TAG_NAME, "img")
    src = img.get_attribute("src")
    return src

page = 1
while True:
    driver.get(LIST_URL.format(page))
    time.sleep(0.8)

    found_any = False
    max_year = 0

    rows = driver.find_elements(
        By.XPATH,
        "//*[@id='contents']/div[3]/div[1]/table/tbody/tr[position()>=1 and position()<=10]"
    )
    if not rows:
        break

    for row in rows:
        # 제목 a
        try:
            title_a = row.find_element(By.CLASS_NAME, "td_tit").find_element(By.TAG_NAME, "a")
            title = title_a.text.strip()
            href  = title_a.get_attribute("href") or ""
        except:
            continue

        # 날짜 td: 작성자(td_write) 다음 형제 td가 날짜
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

        # detail id/URL
        m = DETAIL_HREF_RE.search(href)
        if not m:
            continue
        post_id = m.group(1)
        detail_url = urljoin(BASE, href)

        # 새 탭으로 열고 처리 후 닫기(목록 탭 보존)
        driver.execute_script("window.open(arguments[0]);", detail_url)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(0.5)

        img = get_image()
        # OCR

        text = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        with open(OP_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["구분","세부","평일 운영시간","주말 및 공휴일"])
            writer.writeheader()
            writer.writerows(text)
    else:

        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    # 더 이상 2025가 없고, 페이지 최댓년도가 2024 이하면 종료
    if not found_any and max_year and max_year < 2025:
        break
    page += 1

driver.quit()
