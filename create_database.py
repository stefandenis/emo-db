import os
import shutil


emotions_map = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral"
}

if __name__ == "__main__":
    os.chdir("utterances")
    utterances = os.listdir()
    for utterance in utterances:
        emotion_letter = utterance[5]
        emotion_folder = emotions_map.get(emotion_letter)
        shutil.copy(utterance, '../dataset/' + emotion_folder)
