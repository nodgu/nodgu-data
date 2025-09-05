import os, csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re, time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://www.dongguk.edu"
LIST_URL = BASE + "/article/FOODDFLEX/list?pageIndex={}"
DETAIL_URL = BASE + "/article/FOODDFLEX/detail/{}"

# 정규식
DATE_RE_DOTS = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.$")          # 예: 2025.05.23.
GO_DETAIL_RE = re.compile(r"goDetail\(\s*([0-9]+)\s*\)")

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

OUT_DIR = "output_DFLEX"
os.makedirs(OUT_DIR, exist_ok=True)
OP_CSV = os.path.join(OUT_DIR, "DFLEX.csv")

wait = WebDriverWait(driver, 8)

def norm_date(s: str) -> str:
    """'2025.05.23.' → '2025-05-23' 로 변환"""
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    return s.replace(".", "-")

page = 1
while True:
    MAX_PAGE = 15
    if page > MAX_PAGE: break

    driver.get(LIST_URL.format(page))
    time.sleep(0.8)

    found_any = False
    max_year = 0

    try:
        lis = wait.until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//*[@id='content_focus']/div/div[2]/div[2]/ul/li")
            )
        )
    except:
        lis = []

    if not lis:
        break

    for li in lis:
        try:
            # 앵커(상위에 onclick=goDetail(ID))
            a = li.find_element(By.XPATH, ".//a[contains(@onclick,'goDetail(')]")
            onclick = a.get_attribute("onclick") or ""
            # 제목
            title_el = li.find_element(By.XPATH, ".//p[contains(@class,'tit')]")
            title = title_el.text.strip()
            # 날짜: div.info > span (형식: 2025.05.23.)
            date_el = li.find_element(By.XPATH, ".//div[contains(@class,'info')]/span[1]")
            date_raw = date_el.text.strip()
        except:
            continue

        # 날짜 검사/정규화
        if not DATE_RE_DOTS.fullmatch(date_raw):
            continue
        date_str = norm_date(date_raw)          # 2025-05-23
        year = int(date_str[:4])
        max_year = max(max_year, year)
        if year != 2025:
            continue
        found_any = True

        # 게시글 ID 추출
        m = GO_DETAIL_RE.search(onclick)
        if not m:
            continue
        post_id = m.group(1)
        detail_url = DETAIL_URL.format(post_id)

        # === 새 탭으로 열어 확인 후 닫기(목록 탭 유지) ===
        driver.execute_script("window.open(arguments[0]);", detail_url)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(0.5)

        # 필요 시 본문 첫 이미지 src 확인 (없으면 패스)
        img_src = None
        try:
            img = driver.find_element(By.CSS_SELECTOR, "div.view_cont img")
            img = img.get_attribute("src")
        except:
            pass

        # OCR

        text = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        with open(OP_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["구분","세부","평일 운영시간","주말 및 공휴일"])
            writer.writeheader()
            writer.writerows(text)

        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    # 더 이상 2025 글이 없고, 페이지 최댓년도가 2024 이하면 종료
    if not found_any and max_year and max_year < 2025:
        break
    page += 1

driver.quit()
