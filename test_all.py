import cv2
import json
import os
import numpy as np


# =========================
# PATHS
# =========================
YUNET_PATH = "model/yunet.onnx"
SFACE_PATH = "model/sface.onnx"
EMBEDDINGS_PATH = "model/embeddings.json"
DATASET_PATH = "dataset"


# =========================
# LOAD MODELS
# =========================
detector = cv2.FaceDetectorYN.create(
    YUNET_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_PATH,
    ""
)


# =========================
# LOAD SAVED EMBEDDINGS
# =========================
with open(EMBEDDINGS_PATH, "r") as file:
    student_embeddings = json.load(file)


# =========================
# FUNCTION: GET FACE FEATURE
# =========================
def get_feature(image):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    face = faces[0]

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    feature = recognizer.feature(aligned_face)

    return feature


# =========================
# TEST ALL 5 STUDENTS
# =========================
print("\n==============================")
print("TESTING ALL STUDENT PHOTOS")
print("==============================")

for actual_student in sorted(os.listdir(DATASET_PATH)):

    folder_path = os.path.join(
        DATASET_PATH,
        actual_student
    )

    if not os.path.isdir(folder_path):
        continue

    image_path = os.path.join(
        folder_path,
        "face.jpeg"
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"\n{actual_student} ❌ Image not found")
        continue

    test_feature = get_feature(image)

    if test_feature is None:
        print(f"\n{actual_student} ❌ Face not detected")
        continue

    results = []

    # Compare with every registered student
    for student_id, saved_embedding in student_embeddings.items():

        reference_feature = np.array(
            saved_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        score = recognizer.match(
            reference_feature,
            test_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        results.append(
            (student_id, float(score))
        )

    # Highest similarity first
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_student, best_score = results[0]

    print(f"\nActual Photo : {actual_student}")
    print(f"Predicted    : {best_student}")
    print(f"Similarity   : {best_score:.4f}")

    if actual_student == best_student:
        print("✅ CORRECT MATCH")
    else:
        print("❌ WRONG MATCH")


print("\n==============================")
print("TEST COMPLETED")
print("==============================")