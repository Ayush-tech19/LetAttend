import streamlit as st
import cv2
import json
import numpy as np

from attendance import save_attendance


# =========================================
# PAGE SETTINGS
# =========================================

st.set_page_config(
    page_title="College Attendance System",
    page_icon="🎓",
    layout="wide"
)


# =========================================
# TITLE
# =========================================

st.title("🎓 College Attendance System")

st.write(
    "CNN-Based Face Recognition Attendance System"
)


# =========================================
# FILE PATHS
# =========================================

YUNET_PATH = "model/yunet.onnx"

SFACE_PATH = "model/sface.onnx"

EMBEDDINGS_PATH = "model/embeddings.json"

STUDENTS_PATH = "students.json"


# =========================================
# LOAD STUDENT INFORMATION
# =========================================

with open(
    STUDENTS_PATH,
    "r",
    encoding="utf-8"
) as file:

    students = json.load(file)


# Create dictionary for quick lookup

student_data = {
    student["student_id"]: student
    for student in students
}


# =========================================
# LOAD FACE MODELS
# =========================================

@st.cache_resource
def load_models():

    # YuNet = Face Detection
    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )


    # SFace = Face Recognition
    recognizer = cv2.FaceRecognizerSF.create(
        SFACE_PATH,
        ""
    )


    return detector, recognizer


detector, recognizer = load_models()


# =========================================
# LOAD SAVED EMBEDDINGS
# =========================================

with open(
    EMBEDDINGS_PATH,
    "r"
) as file:

    student_embeddings = json.load(file)


# =========================================
# MATCH THRESHOLD
# =========================================

THRESHOLD = 0.363


# =========================================
# PHOTO UPLOAD
# =========================================

st.subheader("📷 Attendance Photo")

uploaded_file = st.file_uploader(
    "Single ya Group Photo upload karein",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================
# PROCESS PHOTO
# =========================================

if uploaded_file is not None:


    # -------------------------------------
    # Convert uploaded image to OpenCV
    # -------------------------------------

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )


    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "❌ Image load nahi hui."
        )

    else:


        # -------------------------------------
        # Show uploaded photo
        # -------------------------------------

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Attendance Photo",
            use_container_width=True
        )


        # -------------------------------------
        # Get image dimensions
        # -------------------------------------

        height, width = image.shape[:2]


        # -------------------------------------
        # Tell detector image size
        # -------------------------------------

        detector.setInputSize(
            (width, height)
        )


        # -------------------------------------
        # Detect all faces
        # -------------------------------------

        _, faces = detector.detect(
            image
        )


        # =====================================
        # NO FACE
        # =====================================

        if faces is None or len(faces) == 0:

            st.error(
                "❌ Koi face detect nahi hua."
            )


        else:

            st.success(
                f"✅ {len(faces)} face detect hue."
            )


            # =================================
            # IMPORTANT:
            # Set = duplicate students avoid
            # =================================

            recognized_students = set()


            # =================================
            # PROCESS EVERY FACE
            # =================================

            for face_number, face in enumerate(
                faces,
                start=1
            ):


                st.subheader(
                    f"Face {face_number}"
                )


                # ---------------------------------
                # Align detected face
                # ---------------------------------

                aligned_face = recognizer.alignCrop(
                    image,
                    face
                )


                # ---------------------------------
                # Generate embedding
                # ---------------------------------

                test_feature = recognizer.feature(
                    aligned_face
                )


                # ---------------------------------
                # Store comparison results
                # ---------------------------------

                results = []


                # =================================
                # Compare this face with
                # all 5 registered students
                # =================================

                for student_id, saved_embedding in student_embeddings.items():


                    # Convert JSON list back to NumPy
                    reference_feature = np.array(
                        saved_embedding,
                        dtype=np.float32
                    ).reshape(
                        1,
                        -1
                    )


                    # ---------------------------------
                    # Cosine similarity
                    # ---------------------------------

                    score = recognizer.match(
                        reference_feature,
                        test_feature,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )


                    results.append(
                        (
                            student_id,
                            float(score)
                        )
                    )


                # ---------------------------------
                # Highest similarity first
                # ---------------------------------

                results.sort(
                    key=lambda x: x[1],
                    reverse=True
                )


                # ---------------------------------
                # Best match
                # ---------------------------------

                best_student, best_score = results[0]


                # =================================
                # CHECK THRESHOLD
                # =================================

                if best_score >= THRESHOLD:


                    # Get student information
                    data = student_data.get(
                        best_student
                    )


                    if data is not None:


                        # ---------------------------------
                        # Show recognized student
                        # ---------------------------------

                        st.success(
                            f"✅ {data['name']} identified"
                        )


                        st.write(
                            f"**College ID:** "
                            f"{data['college_id']}"
                        )


                        st.write(
                            f"**College:** "
                            f"{data['college_name']}"
                        )


                        st.write(
                            f"**Similarity:** "
                            f"{best_score:.4f}"
                        )


                        # =================================
                        # ADD STUDENT TO PRESENT SET
                        # =================================

                        recognized_students.add(
                            best_student
                        )


                else:


                    # =================================
                    # UNKNOWN FACE
                    # =================================

                    st.warning(
                        f"❌ Unknown person "
                        f"(best score: "
                        f"{best_score:.4f})"
                    )


            # =====================================
            # ATTENDANCE SECTION
            # =====================================

            st.divider()

            st.subheader(
                "📋 Attendance Result"
            )


            # -------------------------------------
            # Show recognized students
            # -------------------------------------

            if len(recognized_students) > 0:

                st.success(
                    f"{len(recognized_students)} "
                    f"registered student(s) recognized."
                )


                for student_id in recognized_students:

                    data = student_data.get(
                        student_id
                    )

                    if data:

                        st.write(
                            f"✅ {data['name']} "
                            f"({data['college_id']})"
                        )


            else:

                st.warning(
                    "Kisi registered student ka "
                    "match nahi mila."
                )


            # =====================================
            # SAVE ATTENDANCE BUTTON
            # =====================================

            if st.button(
                "💾 Save Attendance",
                type="primary"
            ):


                result = save_attendance(
                    students,
                    recognized_students
                )


                if result == "created":

                    st.success(
                        "✅ Attendance Excel me save ho gayi!"
                    )

                else:

                    st.success(
                        "✅ Attendance update ho gayi!"
                    )


                st.info(
                    "📁 File: "
                    "attendance/attendance.xlsx"
                )


# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "Face Detection: YuNet | "
    "Face Recognition: SFace"
)