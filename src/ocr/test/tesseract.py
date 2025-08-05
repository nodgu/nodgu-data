import pytesseract
from dotenv import load_dotenv
from PIL import Image
import requests 
from io import BytesIO

load_dotenv()

print(pytesseract.get_tesseract_version())  # print tesseract-ocr version


url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1724918367312.jpg&contentType=image"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
print(pytesseract.image_to_string(img, lang='kor+eng'))  # print ocr text