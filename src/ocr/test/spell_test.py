import jamspell

def correct_spelling(text):
    # Jamspell 맞춤법 교정기 로드
    corrector = jamspell.TSpellCorrector()
    
    # 한글 사전 로드 (.bin 파일 경로 설정)
    corrector.LoadLangModel('path_to_your_model.bin')
    
    # 맞춤법 교정 수행
    corrected_text = corrector.FixFragment(text)
    
    return corrected_text

# 테스트 문장
input_text = "이 문장은 맞춤법이 틀렷읍니다."

# 맞춤법 교정
corrected_text = correct_spelling(input_text)

print("Original Text: ", input_text)
print("Corrected Text: ", corrected_text)
