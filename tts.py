from gtts import gTTS
from playsound import playsound
import time
import librosa

def ptts():
    text = "화재발생 화재발생 화재발생 화재발생 화재발생"
    tts = gTTS(text=text, lang='ko')
    tts.save("hello_ko.mp3")
    y, sr = librosa.load("hello_ko.mp3")
    time.sleep((len(y)/sr)+1)
    playsound("hello_ko.mp3")

ptts()