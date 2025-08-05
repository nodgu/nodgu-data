import easyocr
import requests 
from io import BytesIO
from PIL import Image
from correction import correct_text_with_et5t2t, correct_text_with_kogpt2, correct_text_with_kogrammar
from gemini_api import gemini_prettier


reader = easyocr.Reader(['ko', 'en'], gpu=False)
custom_dict = ['취업박람회', '쿠폰', '채용상담', '동국대학교', 'dongguk university']

url = "https://www.dongguk.edu/cmmn/fileView?path=/ckeditor//GENERALNOTICES&physical=1724718253381.png&contentType=image"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

img_byte = BytesIO()
img.convert('RGB').save(img_byte, format='JPEG')

result = reader.readtext(img_byte.getvalue())


# for (bbox, text) in result:
#     print(text)

texts = [text for (_, text, _) in result ]
text = '\n'.join(texts)
print(text)

print("==================================")

print(gemini_prettier(text))
# for (bbox, text, confidence) in result:
    # print(f'Text: {text}, Confidence: {confidence}')
    # print("원문: " + text)
    # print(text)
    # print("교정: " + correct_text_with_kogrammar(text))
