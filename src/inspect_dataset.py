from datasets import load_dataset
from collections import Counter


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

print("Loading GoEmotions dataset...")

dataset = load_dataset("google-research-datasets/go_emotions")

print("\nDataset loaded successfully!")


# --------------------------------------------------
# 2. Dataset Information
# --------------------------------------------------

print("\nDataset structure:")
print(dataset)

print("\nNumber of samples:")
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))


# --------------------------------------------------
# 3. Get Emotion Label Names
# --------------------------------------------------

label_feature = dataset["train"].features["labels"].feature
emotion_names = label_feature.names

print("\nTotal emotion categories:", len(emotion_names))

print("\nAll emotion categories:")

for index, emotion in enumerate(emotion_names):
    print(f"{index}: {emotion}")


# --------------------------------------------------
# 4. Required Mood Mentor Emotions
# --------------------------------------------------

target_emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

print("\nMood Mentor target emotions:")

for emotion in target_emotions:
    label_id = emotion_names.index(emotion)
    print(f"{emotion}: {label_id}")


# --------------------------------------------------
# 5. Count Target Emotions
# --------------------------------------------------

target_label_ids = {
    emotion_names.index(emotion): emotion
    for emotion in target_emotions
}


def count_target_emotions(split):

    counter = Counter()

    for example in dataset[split]:

        for label_id in example["labels"]:

            if label_id in target_label_ids:
                emotion = target_label_ids[label_id]
                counter[emotion] += 1

    return counter


print("\n========================================")
print("TARGET EMOTION COUNTS")
print("========================================")

for split in ["train", "validation", "test"]:

    counter = count_target_emotions(split)

    print(f"\n{split.upper()}")

    for emotion in target_emotions:
        print(f"{emotion:10} : {counter[emotion]}")


# --------------------------------------------------
# 6. Check Multi-Label Examples
# --------------------------------------------------

def get_target_emotions(example):

    emotions = []

    for label_id in example["labels"]:

        if label_id in target_label_ids:
            emotions.append(target_label_ids[label_id])

    return emotions


print("\n========================================")
print("MULTI-LABEL AUDIT")
print("========================================")


for split in ["train", "validation", "test"]:

    single_count = 0
    multi_count = 0
    no_target_count = 0

    combination_counter = Counter()

    for example in dataset[split]:

        emotions = get_target_emotions(example)

        if len(emotions) == 0:

            no_target_count += 1

        elif len(emotions) == 1:

            single_count += 1

        else:

            multi_count += 1

            combination = " + ".join(sorted(emotions))

            combination_counter[combination] += 1

    print(f"\n{split.upper()}")

    print("Single target emotion:", single_count)
    print("Multiple target emotions:", multi_count)
    print("No target emotion:", no_target_count)

    print("\nTop emotion combinations:")

    for combination, count in combination_counter.most_common(15):

        print(f"{combination:40} : {count}")


# --------------------------------------------------
# 7. Show Real Multi-Emotion Examples
# --------------------------------------------------

print("\n========================================")
print("SAMPLE MULTI-EMOTION TEXTS")
print("========================================")

shown = 0

for example in dataset["train"]:

    emotions = get_target_emotions(example)

    if len(emotions) >= 2:

        print("\nText:")
        print(example["text"])

        print("Detected dataset emotions:")
        print(", ".join(emotions))

        print("-" * 60)

        shown += 1

        if shown >= 10:
            break


# --------------------------------------------------
# 8. Final Dataset Audit
# --------------------------------------------------

print("\n========================================")
print("FINAL DATASET AUDIT")
print("========================================")

print("\nRequired emotions:")
print(", ".join(target_emotions))

print("\nDataset contains all six required emotions: YES")

print("\nMulti-label capability:")
print("Checked using actual dataset examples.")

print("\nDataset audit completed.")