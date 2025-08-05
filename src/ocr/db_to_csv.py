import psycopg
import csv
import os
from dotenv import load_dotenv


load_dotenv()
db = psycopg.connect(
    host=os.environ["POSTGRES_HOST"],
    dbname=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    port=os.environ["POSTGRES_PORT"]
)

cur = db.cursor()
cur.execute("SELECT * FROM notice_notice WHERE jsonb_array_length(ocr_data) > 0;")

# CSV 파일로 저장
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # 컬럼명 쓰기
    writer.writerow([desc[0] for desc in cur.description])
    # 데이터 쓰기
    writer.writerows(cur)
