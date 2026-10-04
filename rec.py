# import cv2
# import os

# # -----------------------------
# # Model paths
# # -----------------------------
# YUNET_PATH = "model/yunet.onnx"
# SFACE_PATH = "model/sface.onnx"

# # -----------------------------
# # Face Detector
# # -----------------------------
# detector = cv2.FaceDetectorYN.create(
#     YUNET_PATH,
#     "",
#     (320, 320),
#     0.6,
#     0.3,
#     5000
# )

# # -----------------------------
# # Face Recognizer
# # -----------------------------
# recognizer = cv2.FaceRecognizerSF.create(
#     SFACE_PATH,
#     "",
#     cv2.dnn.DNN_BACKEND_OPENCV,
#     cv2.dnn.DNN_TARGET_CPU
# )


# # -----------------------------
# # Function: detect face
# # -----------------------------
# def detect_face(image):

#     height, width = image.shape[:2]

#     detector.setInputSize((width, height))

#     _, faces = detector.detect(image)

#     if faces is None or len(faces) == 0:
#         return None

#     return faces[0]


# # -----------------------------
# # Step 1: Create student
# # reference features
# # -----------------------------
# student_features = {}

# for student_folder in sorted(os.listdir("dataset")):

#     folder_path = os.path.join("dataset", student_folder)

#     if not os.path.isdir(folder_path):
#         continue

#     image_path = os.path.join(folder_path, "face.jpeg")

#     image = cv2.imread(image_path)

#     if image is None:
#         print(student_folder, "❌ Image not found")
#         continue

#     face = detect_face(image)

#     if face is None:
#         print(student_folder, "❌ Face not detected")
#         continue

#     # Align and crop face
#     aligned_face = recognizer.alignCrop(
#         image,
#         face[:-1]
#     )

#     # Extract face features
#     feature = recognizer.feature(aligned_face)

#     student_features[student_folder] = feature

#     print(student_folder, "✅ Reference feature created")


# # -----------------------------
# # Step 2: Load test image
# # -----------------------------
# test_path = "Upload/test.jpg"

# test_image = cv2.imread(test_path)

# if test_image is None:
#     print("\n❌ Upload/test.jpg nahi mili")
#     exit()

# test_face = detect_face(test_image)

# if test_face is None:
#     print("\n❌ Test image me face detect nahi hua")
#     exit()

# # Align and crop
# test_aligned = recognizer.alignCrop(
#     test_image,
#     test_face[:-1]
# )

# # Extract test feature
# test_feature = recognizer.feature(test_aligned)


# # -----------------------------
# # Step 3: Compare test face
# # with every student
# # -----------------------------
# results = []

# for student_id, reference_feature in student_features.items():

#     score = recognizer.match(
#         reference_feature,
#         test_feature,
#         0
#     )

#     results.append((student_id, float(score)))


# # Highest similarity first
# results.sort(key=lambda x: x[1], reverse=True)


# # -----------------------------
# # Step 4: Display results
# # -----------------------------
# print("\n============================")
# print("FACE RECOGNITION RESULTS")
# print("============================")

# for student_id, score in results:
#     print(f"{student_id} : {score:.4f}")

# best_student, best_score = results[0]

# print("\n============================")
# print("BEST MATCH")
# print("============================")

# print("Student:", best_student)
# print("Score:", round(best_score, 4))


import cv2
import json
import numpy as np


# =========================
# MODEL PATHS
# =========================
YUNET_PATH = "model/yunet.onnx"
SFACE_PATH = "model/sface.onnx"
EMBEDDINGS_PATH = "model/embeddings.json"


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
# LOAD TEST IMAGE
# =========================
test_image = cv2.imread("Upload/test.jpg")

if test_image is None:
    print("❌ Upload/test.jpg nahi mili")
    exit()


# =========================
# DETECT FACE
# =========================
height, width = test_image.shape[:2]

detector.setInputSize((width, height))

_, faces = detector.detect(test_image)


if faces is None or len(faces) == 0:
    print("❌ Test image me face nahi mila")
    exit()


print(f"✅ Test image me {len(faces)} face mila")


# =========================
# TEST FACE
# =========================
import numpy as np

THRESHOLD = 0.363

print("\n============================")
print("GROUP FACE RECOGNITION")
print("============================")

for face_number, face in enumerate(faces, start=1):

    # Align detected face
    aligned_face = recognizer.alignCrop(
        test_image,
        face
    )

    # Create embedding
    test_feature = recognizer.feature(
        aligned_face
    )

    results = []

    # Compare with every student
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
        print(
        f"{student_id} -> {score:.4f}"
    )

    # Highest score first
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_student, best_score = results[0]

    print(f"\nFace {face_number}")

    if best_score >= THRESHOLD:

        print("✅ Student:", best_student)
        print("Similarity:", round(best_score, 4))

    else:

        print("❌ Unknown person")
        print("Best score:", round(best_score, 4))