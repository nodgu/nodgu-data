from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

text = """
Creator Crew of Dongguk
크크동 모집
동국대학교 공식 유튜브 크리에이터률 모집합니다
01 지원자격
영상 출연 촬영(소니 미러리스 카메라활용) 편집
(ADOBEPREMIER PRO 프로그램활용) 중1가지 이상
울할 수있는 학생
2개 학기여름 겨울방학 포함) 동안 꾸준한 활동이 가능
한재학생 (휴학생 지원 불가)
정기회의 참석이 가능한 학생(격주 월요일 18시 30분)
직전학기 취특학점 12학점 이상이수 덧 평점평균 2.001
상 취특한 학생
02 우대사항
미러리스카메라와 ADOBE PREMIER PRO틀 함께 다물 수앞는 학생
동국대학교 홍보에 대한아이디어가 풍부한 학생
03 활동혜택
굳렌- 제작 건당 장학금 지급 (통9소픔 차등있음)
촬영 장비 및 편집 프로그램 지원
활동완료 후 수로증 지급
04모집절차
1차: 서류전형(자기소개서 제출)
~2024.09.03.
*철부되 지원서 작성 후E-MAIL 접수
(leeray@donggukedu)
2차: 면접 전형대면 진행)
2024.09.06.
오리언테이선 일정
2024.09.09.18.30
"""

prompt = f"{text} 이거 OCR에서 얻은 글인데 잘못된 글자도 수정하고 보기 좋게 만들어봐"

prompt = input("프롬프트를 입력하세요: ")

input_ids = tokenizer.encode(prompt, return_tensors="pt")

gen_ids = model.generate(input_ids,
                        #    max_length=len(prompt),
                           max_length = 256,
                        #    max_new_tokens=512,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

generated = tokenizer.decode(gen_ids[0])

print(generated)