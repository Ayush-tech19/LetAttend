import cv2
import os
import json


# =========================
# MODEL PATHS
# =========================
YUNET_PATH = "model/yunet.onnx"
SFACE_PATH = "model/sface.onnx"
OUTPUT_PATH = "model/embeddings.json"


# =========================
# FACE DETECTOR
# =========================
detector = cv2.FaceDetectorYN.create(
    YUNET_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# =========================
# FACE RECOGNIZER
# =========================
recognizer = cv2.FaceRecognizerSF.create(
    SFACE_PATH,
    ""
)


# =========================
# EMPTY DICTIONARY
# =========================
embeddings = {}


# =========================
# READ ALL STUDENTS
# =========================
for student_folder in sorted(os.listdir("dataset")):

    folder_path = os.path.join("dataset", student_folder)

    if not os.path.isdir(folder_path):
        continue

    image_path = os.path.join(folder_path, "face.jpeg")

    image = cv2.imread(image_path)

    if image is None:
        print(f"{student_folder} ❌ Image nahi mili")
        continue

    # Image size
    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    # Detect face
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        print(f"{student_folder} ❌ Face detect nahi hua")
        continue

    # First detected face
    face = faces[0]

    # Align face
    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    # Create embedding
    feature = recognizer.feature(aligned_face)

    # Convert numpy array to normal Python list
    feature_list = feature.flatten().tolist()

    # Save embedding
    embeddings[student_folder] = feature_list

    print(f"{student_folder} ✅ Embedding created")


# =========================
# SAVE TO JSON
# =========================
with open(OUTPUT_PATH, "w") as file:

    json.dump(embeddings, file)

print("\n==============================")
print("ALL EMBEDDINGS SAVED")
print("==============================")
print(f"Saved in: {OUTPUT_PATH}")