from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os, re, pandas as pd

BASE = "https://dgucoop.dongguk.edu/mobile/menu_list.html"

# ===== Selenium 설정 =====
options = Options()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 8)

# ===== 상수 =====
PRICE_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d{3,5})\s*원")
FLOOR_LABELS = [
    ("상록원1층", "상록원1층"),
    ("상록원2층", "상록원2층"),
    ("상록원3층", "상록원3층"),
]
DAY_ORDER = ["일","월","화","수","목","금","토"]
DAY_XPATHS = [
    "/html/body/div[1]/div[2]/ul/li[2]/div[1]",   # 일
    "/html/body/div[1]/div[2]/ul/li[2]/div[3]",   # 월
    "/html/body/div[1]/div[2]/ul/li[2]/div[5]",   # 화
    "/html/body/div[1]/div[2]/ul/li[2]/div[7]",   # 수
    "/html/body/div[1]/div[2]/ul/li[2]/div[9]",   # 목
    "/html/body/div[1]/div[2]/ul/li[2]/div[11]",  # 금
    "/html/body/div[1]/div[2]/ul/li[2]/div[13]",  # 토
]

# ===== 유틸 =====
def split_menu_price(line: str):
    s = line.strip()
    price = ""
    m_all = list(PRICE_RE.finditer(s))
    if m_all:
        m = m_all[-1]
        price = m.group(1).replace(",", "")
        s = (s[:m.start()] + s[m.end():]).strip()
    s = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;·")
    return s, price

def parse_table(floor_name: str, day_label: str):
    rows = []
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//table//tr")))
    except:
        return rows

    trs = driver.find_elements(By.XPATH, "//table//tr")
    if not trs:
        return rows

    ths = trs[0].find_elements(By.XPATH, "./td|./th")
    if len(ths) < 2:
        return rows
    meal_names = [(td.text or "").strip() or "중식" for td in ths[1:]]

    for tr in trs[1:]:
        tds = tr.find_elements(By.XPATH, "./td")
        if len(tds) < 2: continue
        corner = (tds[0].text or "").strip() or "메뉴"

        for j, meal in enumerate(meal_names, start=1):
            if j >= len(tds): continue
            cell = (tds[j].text or "").strip()
            if not cell: continue

            for ln in [x.strip() for x in cell.splitlines() if x.strip()]:
                if re.search(r"\d{1,2}:\d{2}\s*~\s*\d{1,2}:\d{2}", ln):
                    continue
                menu, price = split_menu_price(ln)
                if not menu and not price: continue
                rows.append([day_label, floor_name, corner, meal, menu, price])
    return rows

# ===== 실행 =====
driver.get(BASE)
wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
time.sleep(0.5)

all_rows = []

for label_sub, floor_name in FLOOR_LABELS:
    links = driver.find_elements(By.XPATH, f"//a[contains(normalize-space(.), '{label_sub}')]")
    if not links:
        print(f"[WARN] '{label_sub}' 링크 못찾음")
        continue

    # 층 탭 열기
    driver.execute_script("window.open(arguments[0]);", links[0].get_attribute("href"))
    driver.switch_to.window(driver.window_handles[-1])
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(0.5)

    # 일~토 요일 버튼 클릭 순회
    for day_label, xp in zip(DAY_ORDER, DAY_XPATHS):
        try:
            btn = wait.until(EC.element_to_be_clickable((By.XPATH, xp)))
            btn.click()
            time.sleep(0.8)  # 내용 로딩 대기
            all_rows.extend(parse_table(floor_name, day_label))
            print(f"[INFO] {floor_name} {day_label} 저장완료")
        except Exception as e:
            print(f"[WARN] {floor_name} {day_label}: {e}")
            continue

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

driver.quit()

# ===== 저장 =====
os.makedirs("output_SANGNOK", exist_ok=True)
df = pd.DataFrame(all_rows, columns=["요일","층","코너","식사","메뉴","가격"])
if not df.empty:
    cat = pd.CategoricalDtype(categories=DAY_ORDER, ordered=True)
    df["요일"] = df["요일"].astype(cat)
    df = df.sort_values(["요일","층","코너","식사","메뉴"]).reset_index(drop=True)
out_path = "output_SANGNOK/상록원_1~3층_주간메뉴(요일탭클릭).csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[저장 완료] {out_path}")
