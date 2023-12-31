# -*- coding: utf-8 -*-
"""gradio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_GWRGRVb3znpFoAUO_7jj36vZG2FseJi
"""

# !pip install gradio
# !pip install wit
# !pip install scipy

import gradio as gr
from wit import Wit
import numpy as np
import io
from scipy.io.wavfile import write

def is_enrolled_voice(name, wave_file):
    return True

def STT(wave_file):
    # Wit.ai API 연결을 위한 클라이언트 생성
    wit_api_key = "HBAX6TJEDWRKUIUA62MFQZCXYG4BKQBQ"
    wit_url = "https://api.wit.ai/speech"
    wit_client = Wit(wit_api_key)

    # 음성 파일에서 텍스트 추출
    print("STT 수행 전")
    print("wave_file 의 타입 :", type(wave_file))
    print(wave_file)

    sample_rate, data = wave_file
    # 정규화 및 16비트로 변환
    scaled_data = (data / np.max(np.abs(data)) * 32767).astype(np.int16)
    # 변수에 .wav 파일 데이터 저장
    wav_data = io.BytesIO()
    write(wav_data, sample_rate, scaled_data)

    response = wit_client.speech(wav_data, headers={'Content-Type': 'audio/wav'})
    print("STT 수행 후")
    print(response)
    print(type(response))
    # 추출된 텍스트 확인
    if 'text' in response:
        return response['text']
    else:
        return "추출된 텍스트 없음!"

def function(name, wave_file):
    is_enrolled = is_enrolled_voice(name, wave_file)
    if is_enrolled:  # 등록된 목소리인 경우 STT
        print("등록된 목소리 입니다!")
        output_text = STT(wave_file)
        return output_text
    else:
        return "등록되지 않은 목소리입니다!"

def main():
    demo = gr.Interface(
        function,
        inputs=["text", "audio"],
        outputs=["text"]
    )
    demo.launch(debug=True, share=True)

if __name__ == '__main__':
    main()