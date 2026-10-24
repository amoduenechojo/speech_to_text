import speech_recognition as sr
from speech_recognition import UnknownValueError

recognizer = sr.Recognizer()


def record_text():
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source)
                my_text = recognizer.recognize_google(audio, language='en-GB')
                return my_text

        except sr.RequestError as error:
            print("could not request results from Google Speech Recognition service: {0}".format(error))

        except UnknownValueError:
            print("could not understand audio")


def output_text(text):
    with open("output.txt", "a") as f:
        f.write(text)
        f.write("\n")


if __name__ == "__main__":
    while True:
        spoken_text = record_text()
        output_text(spoken_text)
        print("Wrote text to output.txt")