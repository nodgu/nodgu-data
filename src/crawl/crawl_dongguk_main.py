import requests
from bs4 import BeautifulSoup
import json
from pprint import pprint
from datetime import date, datetime
from dotenv import load_dotenv
import os

# .env 파일 load
load_dotenv()
API_HOST = os.getenv("API_HOST")


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


def crawl_main_notices():
    # notitypes = "JANGHAKNOTICE GENERALNOTICES HAKSANOTICE IPSINOTICE GLOBALNOLTICE HAKSULNOTICE SAFENOTICE BUDDHISTEVENT".split(
    #     " "
    # )
    notitypes = ["GENERALNOTICES", "JANGHAKNOTICE", "HAKSANOTICE", "IPSINOTICE", "GLOBALNOLTICE", "HAKSULNOTICE", "SAFENOTICE", "BUDDHISTEVENT"]

    # pages = {
    #     "GENERALNOTICES": 88,
    #     "IPSINOTICE": 10,
    #     "HAKSANOTICE": 56,
    #     "GLOBALNOLTICE": 42,
    #     "HAKSULNOTICE": 5,
    #     "BUDDHISTEVENT": 12,
    #     "JANGHAKNOTICE": 54,
    #     "SAFENOTICE": 3,
    # }

    pages = {
        "GENERALNOTICES": 2,
        "IPSINOTICE": 2,
        "HAKSANOTICE": 2,
        "GLOBALNOLTICE": 2,
        "HAKSULNOTICE": 2,
        "BUDDHISTEVENT": 2,
        "JANGHAKNOTICE": 2,
        "SAFENOTICE": 2,
    }

    for notitype in notitypes:  # 종류별로 for문 돌리기
        for i in range(1, pages[notitype]):
            url = f"https://www.dongguk.edu/article/{notitype}/list?pageIndex=1"
            base_url = "https://www.dongguk.edu"
            print(f"{notitype}")
            print(f"Fetching Url: {url}")
            response = requests.get(url)
            notices_list = []

            if response.status_code == 200:
                html = response.text  # 받아온 것의 텍스트
                soup = BeautifulSoup(html, "html.parser")  # 텍스트 해석??
                noti_list = soup.select_one("div.board_list > ul")
                notices = noti_list.select("li") 
                for notice in notices:  # 각 공지별로 for문 돌리기
                    title = (
                        notice.select_one("p.tit")
                        .get_text()
                        .replace("\t", "")
                        .replace("\r\n", "")
                        .replace("\n", "")
                    )
                    if title.startswith("\n공지\r\n"):
                        continue
                        
                    print(f"Title: {title}")
                    id = (
                        notice.select_one("a")["onclick"]
                        .replace("goDetail(", "")
                        .replace(");", "")
                    )
                    notice_url = f"https://www.dongguk.edu/article/{notitype}/detail/{id}"
                    # 새 url 접속
                    print(f"Fetching Url: {notice_url}")
                    response_detail = requests.get(notice_url)
                    soup_detail = BeautifulSoup(response_detail.text, "html.parser")

                    description_html = soup_detail.select_one(
                        "#content_focus > div > div.board_view > div.view_cont"
                    )
                    if description_html is None:
                        print(f"ERROR: Can't find #content_focus > div > div.board_view > div.view_cont in {notice_url}")
                        break
                    for s in description_html.select("script"):
                        s.extract()
                    date_html = soup_detail.select_one(
                        "#content_focus > div > div.board_view > div.tit > div > span:nth-child(1)"
                    )
                    notice_date_str = (
                        date_html
                        .get_text()
                        .replace("등록일 ", "")
                    )
                    # 날짜 파싱 개선
                    try:
                        ymds = notice_date_str.split('.')[:-1]
                        ymd = [int(x) for x in ymds]
                        notice_date = date(ymd[0], ymd[1], ymd[2])
                    except (ValueError, IndexError) as e:
                        print(f"날짜 파싱 에러: {notice_date_str}, {e}")
                        notice_date = date.today()  # 기본값으로 오늘 날짜 사용
                    # author =   작성자 #content_focus > div > div.board_view > div.tit > div > span:nth-child(2)
                    # views =    조회수 #content_focus > div > div.board_view > div.tit > div > span:nth-child(3)

                    ps = soup_detail.select(
                        "#content_focus > div > div.board_view > div.view_cont > p"
                    )
                    imgs = []
                    for p in ps:
                        imgsS = p.select("img")
                        for img in imgsS:
                            v = img.get("src", img.get("dfr-src"))
                            if v is None:
                                continue
                            # 상대 경로를 절대 경로로 변환
                            if v.startswith('/'):
                                imgs.append(f"{base_url}{v}")
                            elif v.startswith('http'):
                                imgs.append(v)
                            else:
                                imgs.append(f"{base_url}/{v}")

                    # 링크 처리 개선
                    links = []
                    for p in ps:
                        linksS = p.select("a")
                        for link in linksS:
                            href = link.get("href")
                            if href is None:
                                continue
                            # 상대 경로를 절대 경로로 변환
                            if href.startswith('/'):
                                full_url = f"{base_url}{href}"
                            elif href.startswith('http'):
                                full_url = href
                            else:
                                full_url = f"{base_url}/{href}"
                            links.append({"name": link.get_text(), "url": full_url})

                    # 첨부파일(지원서 양식 등)
                    attachments = []
                    view_files = soup_detail.select_one(
                        "#content_focus > div > div.board_view > div.view_files > ul"
                    )
                    # location.href="/cmmn/fileDown.do?filename="+encodeURIComponent(file_nm)+"&filepath="+file_path+"&filerealname="+file_sys_nm;

                    if view_files:
                        attachment_elements = view_files.select("li > a")
                        for attachment in attachment_elements:
                            href: str = attachment["href"]
                            file_url = None
                            down_file = ""
                            if href.startswith("javascript:downGO("):
                                parts = href[len("javascript:downGO(") : -1].split("','")
                                if len(parts) >= 3:
                                    temp = parts[2].replace("')", "")
                                    down_file = f"/cmmn/fileDown.do?filename={parts[0]}&filepath={parts[1]}&filerealname={temp}"
                            if down_file != "":
                                attachments.append(
                                    {
                                        "name": attachment.get_text(),
                                        "url": base_url + down_file,
                                    }
                                )

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


                    # try: 
                    #     Notice.objects.get(
                    #         notice_id=id,
                    #         univ_code="DONGGUK",
                    #         org_code="MAIN",
                    #         sub_code=notitype,)
                    # except Notice.DoesNotExist:
                    #     notice_object = Notice(**notice_data)
                    #     notice_object.save()
                    #     notices_list.append(notice_data)
                    #     print(f"Save to DB: {notice_url}\n")
                    print(f"Complete: {notice_url}")
                open(f"./data/{notitype}.json", "w", encoding="UTF-8").write(
                    json.dumps(notices_list, ensure_ascii=False, default=json_serial)
                )
                print(f"Complete: {url}")
            else:
                pprint(response)


if __name__ == "__main__":
    crawl_main_notices()
