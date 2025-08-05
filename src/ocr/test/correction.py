from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from symspellpy_ko import KoSymSpell, Verbosity


def correct_text_with_kogpt2(text):
    # 사전 학습된 KoGPT-2 모델과 토크나이저 불러오기
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>')
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    # 입력 문장을 토큰화
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=len(text)+5)

    # 모델을 통해 문맥 기반 텍스트 생성
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # 토큰을 다시 문장으로 변환
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

    # # KoGPT-2를 통해 문맥 교정
    # corrected_text = correct_text_with_kogpt2(input_text)
    # print("Original Text: ", input_text)
    # print("Corrected Text: ", corrected_text)


def correct_text_with_symspell_ko(text):
    sym_spell = KoSymSpell()
    sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)

    print(f"\n\n원문: {text}\n")
    for suggestion in sym_spell.lookup(text, Verbosity.ALL):
        print("", suggestion.term, suggestion.distance, suggestion.count)


def correct_text_with_et5t2t(text):
    # T5 모델 로드
    model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
    tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "mps:0" if torch.cuda.is_available() else "cpu" # for mac m1
    device = "cpu"
    model = model.to(device) 

    # 예시 입력 문장
    # input_text = "아늬 진짜 무ㅓ하냐고"
    input_text = text

    # 입력 문장 인코딩
    input_encoding = tokenizer("맞춤법을 고쳐주세요: " + input_text, return_tensors="pt")

    input_ids = input_encoding.input_ids.to(device)
    attention_mask = input_encoding.attention_mask.to(device)

    # T5 모델 출력 생성
    output_encoding = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )

    # 출력 문장 디코딩
    output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)

    # 결과 출력
    print(f"원 문: {text}")
    print(f"교정됨: {output_text}") # 아니 진짜 뭐 하냐고.


def correct_text_with_kogrammar(text):

    # theSOL1/kogrammar-distil
    tokenizer = AutoTokenizer.from_pretrained("theSOL1/kogrammar-distil")
    model = AutoModelForSeq2SeqLM.from_pretrained("theSOL1/kogrammar-distil")

    # 입력 텍스트를 토큰화 (입력 형태에 맞게)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, return_token_type_ids=False)

    # 모델을 통해 문법 및 오타 교정
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)

    # 토큰화된 결과를 텍스트로 변환
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text