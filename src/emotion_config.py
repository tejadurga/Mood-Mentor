# Mood Mentor - Emotion Configuration

EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

LABEL2ID = {
    emotion: index
    for index, emotion in enumerate(EMOTION_LABELS)
}

ID2LABEL = {
    index: emotion
    for index, emotion in enumerate(EMOTION_LABELS)
}

NUM_LABELS = len(EMOTION_LABELS)