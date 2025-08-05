import easyocr
import requests
from io import BytesIO
from PIL import Image
# from correction import correct_text_with_et5t2t, correct_text_with_kogpt2, correct_text_with_kogrammar
from test.gemini_api import gemini_prettier
import psycopg
from dotenv import load_dotenv
import os
import json

def ocr(url: str):
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    custom_dict = ['취업박람회', '쿠폰', '채용상담', '동국대학교', 'dongguk university']

    # url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1724718253381.png&contentType=image"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    img_byte = BytesIO()
    img.convert('RGB').save(img_byte, format='JPEG')

    result = reader.readtext(img_byte.getvalue())


    # for (bbox, text) in result:
    #     print(text)

    texts = [text for (_, text, _) in result ]
    text = '\n'.join(texts)
    # print(text)

    # print("==================================")

    return gemini_prettier(text)



def get_data():
    load_dotenv()

    db = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        port=os.environ["POSTGRES_PORT"]
    )

    cursor = db.cursor()
    cursor.execute("SELECT id, imgs, tdindex FROM notice_notice WHERE jsonb_array_length(ocr_data) = 0 AND jsonb_array_length(imgs) > 0 LIMIT 3;")

    for (id, imgs, tdindex) in cursor.fetchall():
        print(id, imgs)

        ocr_data = []
        for img in imgs:
            print(img)
            ocr_data.append(ocr(img))
        new_tdindex = tdindex + "\n" + "\n".join(ocr_data)
        cursor.execute("UPDATE notice_notice SET ocr_data = %s, tdindex = %s WHERE id = %s;", (json.dumps(ocr_data), new_tdindex, id))
        db.commit()
    cursor.close()


if __name__ == "__main__":
    get_data()
