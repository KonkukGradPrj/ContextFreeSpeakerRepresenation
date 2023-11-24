from wit import Wit
import os

import pyaudio
import wave
import os
from datetime import datetime


class Recorder:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 10  # 녹음 시간 (초)
    OUTPUT_DIR = "records"  # 음성 파일 저장 디렉토리

    def __init__(self, dir) -> None:
        # 녹음 경로 설정
        self.OUTPUT_DIR = "records"


    def start_record(self):
        # ctrl c를 눌러 녹음 종료
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        # 녹음 파일 이름 생성
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = os.path.join(self.OUTPUT_DIR, f"{current_time}.wav")

        # PyAudio 초기화
        audio = pyaudio.PyAudio()

        # 녹음 스트림 열기
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)

        print("녹음 시작...")
        frames = []

        # 녹음 진행
        try:
            while(True):
                data = stream.read(self.CHUNK)
                frames.append(data)

        except KeyboardInterrupt:
            # ctrl c를 눌러 녹음 종료
            print("녹음 종료.\n")

        # 녹음 스트림 닫기
        stream.stop_stream()
        stream.close()

        # PyAudio 종료
        audio.terminate()

        # 녹음 파일 저장
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))


def wit_transcribe_audio(api_key, audio_file_path):
    # Wit.ai API 연결을 위한 클라이언트 생성
    wit_client = Wit(api_key)

    # 음성 파일에서 텍스트 추출
    with open(audio_file_path, 'rb') as audio_file:
        response = wit_client.speech(audio_file, headers={'Content-Type': 'audio/wav'})

    # 추출된 텍스트 확인
    if 'text' in response:
        return response['text']
    else:
        return None

def STT(dir):
    audio_files = [f for f in os.listdir(dir) if f.endswith('.wav') or f.endswith('.mp3')]

    for audio_file in audio_files:
        # 음성 파일의 전체 경로
        audio_file_path = os.path.join('records', audio_file)

        # Wit.ai를 사용하여 텍스트 추출
        transcribed_text = wit_transcribe_audio(wit_api_key, audio_file_path)

        # 추출된 텍스트를 CSV 파일로 저장
        csv_file_path = os.path.splitext(audio_file_path)[0] + '.csv'
        with open(csv_file_path, 'w', encoding='utf-8') as csv_file:
            csv_file.write(f'Time,Text\n0:00,{transcribed_text}')

        print(f'Transcription saved to {csv_file_path}')



if __name__ == "__main__":
    # Wit.ai API 키
    wit_api_key = "HBAX6TJEDWRKUIUA62MFQZCXYG4BKQBQ"
    path = 'records'
    rec = Recorder(path)
    try:
        while(True):
            print("1. 녹음하기\n2. 텍스트 변환하기\n3. 종료하기")
            menu = input("메뉴를 입력하세요 : ")
            if menu not in ['1','2','3']:
                print("잘못된 입력입니다.\n")
                continue
            elif menu == '1':
                rec.start_record()
            elif menu == '2':
                STT(path)
            elif menu == '3':
                print("프로그램을 종료합니다.")
                break
    except KeyboardInterrupt:
        print("프로그램 종료!")
    except:
        print("오류발생!")




