from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
import time
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
    return week_date_int

base_url = "https://dgucoop.dongguk.edu"
url = base_url + "/store/store.php?w=4&l=2&j={}"

notitypes = ["SANGNOK1", "SANGNOK2", "SANGNOK3", "BMC", "DFLEX"]
pages = {
    "SANGNOK1": 1,
    "SANGNOK2": 2,
    "SANGNOK3": 3,
    "BMC": 4,
    "DFLEX": 5,
}

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

idx = 0
driver.get(url.format(idx))
current_week = driver.find_element(By.CLASS_NAME, "menu_date").text.strip()
current_week_date = to_int(current_week)
last_week_date = 20090830


# 1. null일 때 null 반환시키기
# 2. 가격 span 의존성. span[1]이 비어있지 않을 때만 가격 가져오기
# 3. 세부 내용(시간이나 돼지 수입산인지 이런거나 가격도 이상하게 되어있기도 함)
while True:
    response = requests.get(url)
    menu_list = []

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        current_week = soup.find_element(By.CLASS_NAME, "menu_date")
        current_week_date = to_int(current_week)
        if (current_week_date >= last_week_date):
            break
        
        for i in range(2, 9):
            today = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[1]/td[{i}]/span/text()[2]')

            sangnok3_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[2]/td')
            sangnok3_cat1_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[3]/td[1]')
            sangnok3_cat1_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[3]/td[{i+1}]/span[1]')
            sangnok3_cat1_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[3]/td[{i+1}]/span[2]')
            sangnok3_cat1_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[4]/td[{i}]/span[1]')
            sangnok3_cat1_dinner_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[4]/td[{i}]/span[2]')
            
            sangnok3_cat2_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[5]/td[1]')
            sangnok3_cat2_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[5]/td[{i+1}]/span[1]')
            sangnok3_cat2_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[5]/td[{i+1}]/span[2]')
            sangnok3_cat2_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[6]/td[{i}]/span[1]')
            sangnok3_cat2_dinner2_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[6]/td[{i}]/span[2]')


            sangnok2_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[7]/td')
            sangnok2_cat1_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[8]/td[1]')
            sangnok2_cat1_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[8]/td[{i+1}]/span[1]')
            sangnok2_cat1_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[8]/td[{i+1}]/span[2]')
            sangnok2_cat1_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[9]/td[{i}]/span[1]')
            sangnok2_cat1_dinner_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[9]/td[{i}]/span[2]')
            
            sangnok2_cat2_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[10]/td[1]')
            sangnok2_cat2_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[10]/td[{i+1}]/span[1]')
            sangnok2_cat2_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[10]/td[{i+1}]/span[2]')
            sangnok2_cat2_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[11]/td[{i}]/span[1]')
            sangnok2_cat2_dinner2_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[11]/td[{i}]/span[2]')

            sangnok2_cat3_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[11]/td[1]')
            sangnok2_cat3_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[11]/td[{i+1}]/span[1]')
            sangnok2_cat3_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[11]/td[{i+1}]/span[2]')
            sangnok2_cat3_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[12]/td[{i}]/span[1]')
            sangnok2_cat3_dinner2_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[12]/td[{i}]/span[2]')

            sangnok2_cat4_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[13]/td[1]')
            sangnok2_cat4_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[13]/td[{i+1}]/span[1]')
            sangnok2_cat4_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[13]/td[{i+1}]/span[2]')
            sangnok2_cat4_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[14]/td[{i}]/span[1]')
            sangnok2_cat4_dinner2_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[14]/td[{i}]/span[2]')


            sangnok1_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[15]/td')
            sangnok1_cat1_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[16]/td[1]')
            sangnok1_cat1 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[16]/td[{i+1}]/span[1]')
            sangnok1_cat1_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[16]/td[{i+1}]/span[2]')

            sangnok1_cat2_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[18]/td[1]')
            sangnok1_cat2 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[18]/td[{i+1}]/span[1]')
            sangnok1_cat2_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[18]/td[{i+1}]/span[2]')

            sangnok1_cat3_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[19]/td[1]')
            sangnok1_cat3 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[19]/td[{i+1}]/span[1]')
            sangnok1_cat3_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[19]/td[{i+1}]/span[2]')

            sangnok1_cat4_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[20]/td[1]')
            sangnok1_cat4 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[20]/td[{i+1}]/span[1]')
            sangnok1_cat4_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[20]/td[{i+1}]/span[2]')

            sangnok1_cat5_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[21]/td[1]')
            sangnok1_cat5 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[21]/td[{i+1}]/span[1]')
            sangnok1_cat5_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[21]/td[{i+1}]/span[2]')

            sangnok1_cat6_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[22]/td[1]')
            sangnok1_cat6 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[22]/td[{i+1}]/span[1]')
            sangnok1_cat6_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[22]/td[{i+1}]/span[2]')

            sangnok1_cat7_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[23]/td[1]')
            sangnok1_cat7 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[23]/td[{i+1}]/span[1]')
            sangnok1_cat7_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[23]/td[{i+1}]/span[2]')


            bmc_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[24]/td')
            bmc_breakfast_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[25]/td[1]')
            bmc_breakfast = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[25]/td[{i}]/span[1]')
            bmc_breakfast_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[25]/td[{i}]/span[2]')
            
            bmc_cat1_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[26]/td[1]')
            bmc_cat2_lunch = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[26]/td[{i+1}]/span[1]')
            bmc_cat2_lunch_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[26]/td[{i+1}]/span[2]')
            bmc_cat1_dinner = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[27]/td[{i}]/span[1]')
            bmc_cat1_dinner_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[27]/td[{i}]/span[2]')
            
            bmc_cat3_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[28]/td[1]')
            bmc_cat3 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[28]/td[{i}]/span[1]')
            bmc_cat3_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[28]/td[{i}]/span[2]')

            bmc_cat4_name = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[29]/td[1]')
            bmc_cat4 = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[29]/td[{i}]/span[1]')
            bmc_cat4_cost = soup.find_element(By.XPATH, f'//*[@id="sdetail"]/table[2]/tbody/tr[29]/td[{i}]/span[2]')


            notice_data = {
                "noticeId": id,
                "title": title,
                "url": notice_url,
                "description": description_html.prettify(),
                "notitype": notitype,
                "date": notice_date.isoformat() + "T00:00:00",  # LocalDateTime 형식으로 변환
                "tdindex": title + description_html.get_text().replace("\n", ""),
                "imgs": imgs,
                "links": [link["url"] for link in links],  # URL만 추출하여 List<String>으로 변환
                "attachments": attachments,
                "ocrData": "",  # String으로 변환
                "univCode": "DONGGUK",
                "orgCode": "MAIN",
                "subCode": notitype
            }
            
            try:
                response = requests.post(f"{API_HOST}/api/v1/notice/notice", json=notice_data, timeout=10)
                if response.status_code == 200:
                    print(f"Save to DB: {notice_url}\n")
                else:
                    print(f"API 에러 - 상태코드: {response.status_code}, 응답: {response.text}")
                    print(f"요청 데이터: {notice_data}")
            except requests.exceptions.ConnectionError:
                print(f"연결 에러: Spring 서버가 실행 중인지 확인하세요 ({API_HOST})")
            except requests.exceptions.Timeout:
                print(f"타임아웃 에러: {notice_url}")
            except Exception as e:
                print(f"예상치 못한 에러: {e}")
                print(f"에러 발생 데이터: {notice_data}")

            print(f"Complete: {notice_url}")
        open(f"./data/{notitype}.json", "w", encoding="UTF-8").write(
            json.dumps(menu_list, ensure_ascii=False, default=json_serial)
        )
        print(f"Complete: {url}")
    else:
        pprint(response)
