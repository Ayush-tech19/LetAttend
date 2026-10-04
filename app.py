import os
import json
import cv2
import numpy as np
import streamlit as st
from datetime import date
from openpyxl import Workbook, load_workbook


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
STUDENT_FACES_DIR = os.path.join(BASE_DIR, "student_faces")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")

YUNET_PATH = os.path.join(MODEL_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "sface.onnx")

STUDENTS_FILE = os.path.join(DATABASE_DIR, "students.json")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "embeddings.json")

EXCEL_FILE = os.path.join(
    ATTENDANCE_DIR,
    "Attendance.xlsx"
)


# ============================================================
# SETTINGS
# ============================================================

# Strict enough to reduce false positives
RECOGNITION_THRESHOLD = 0.50

MIN_REGISTRATION_PHOTOS = 3
MAX_REGISTRATION_PHOTOS = 5


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="College Attendance System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STUDENT_FACES_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ============================================================
# DEFAULT FILES
# ============================================================

if not os.path.exists(STUDENTS_FILE):

    with open(
        STUDENTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump([], f, indent=4)


if not os.path.exists(EMBEDDINGS_FILE):

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({}, f, indent=4)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(
            f"Missing model: {YUNET_PATH}"
        )

    if not os.path.exists(SFACE_PATH):
        raise FileNotFoundError(
            f"Missing model: {SFACE_PATH}"
        )

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

    return detector, recognizer


try:

    detector, recognizer = load_models()

except Exception as e:

    st.error("Face recognition models could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# STUDENT FUNCTIONS
# ============================================================

def load_students():

    try:

        with open(
            STUDENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception:
        return []


def save_students(students):

    with open(
        STUDENTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            students,
            f,
            indent=4,
            ensure_ascii=False
        )


def load_embeddings():

    try:

        with open(
            EMBEDDINGS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, dict):
                return data

            return {}

    except Exception:
        return {}


def save_embeddings(embeddings):

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            embeddings,
            f,
            indent=4
        )


# ============================================================
# IMAGE HELPERS
# ============================================================

def bytes_to_image(file_bytes):

    array = np.asarray(
        bytearray(file_bytes),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return image


def detect_faces(image):

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    if faces is None:
        return []

    return list(faces)


# ============================================================
# CREATE EMBEDDINGS FOR REGISTRATION
# ============================================================

def create_embedding(image):

    faces = detect_faces(image)

    if len(faces) == 0:
        return None, "No face detected."

    if len(faces) > 1:
        return None, "Multiple faces detected. Use a photo containing only one person."

    face = faces[0]

    try:

        aligned = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned
        )

        feature = feature.flatten().astype(
            np.float32
        )

        return feature.tolist(), None

    except Exception as e:

        return None, str(e)


# ============================================================
# RECOGNITION
# ============================================================

def normalize_embedding(value):

    arr = np.array(
        value,
        dtype=np.float32
    )

    # Old format: single embedding
    if arr.ndim == 1:

        return [arr.reshape(1, -1)]

    # Multiple embeddings
    if arr.ndim == 2:

        return [
            x.reshape(1, -1)
            for x in arr
        ]

    return []


def recognize_face(
    image,
    face,
    embeddings
):

    try:

        aligned = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned
        )

    except Exception:

        return None, 0.0


    best_student = None
    best_score = -1.0


    for student_id, stored in embeddings.items():

        references = normalize_embedding(
            stored
        )

        for reference in references:

            try:

                score = recognizer.match(
                    feature,
                    reference,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:

                    best_score = score
                    best_student = student_id

            except Exception:
                continue


    if (
        best_student is not None
        and best_score >= RECOGNITION_THRESHOLD
    ):

        return best_student, best_score


    return None, best_score


# ============================================================
# EXCEL
# ============================================================

def create_excel(students):

    if os.path.exists(EXCEL_FILE):
        return

    wb = Workbook()

    ws = wb.active
    ws.title = "Attendance"

    headers = [
        "College ID",
        "Name",
        "Branch",
        "Year",
        "Section"
    ]

    ws.append(headers)

    for student in students:

        ws.append([
            student.get("college_id", ""),
            student.get("name", ""),
            student.get("branch", ""),
            student.get("year", ""),
            student.get("section", "")
        ])

    wb.save(EXCEL_FILE)


def sync_students_to_excel(students):

    create_excel(students)

    wb = load_workbook(
        EXCEL_FILE
    )

    ws = wb["Attendance"]

    existing_ids = {}

    for row in range(
        2,
        ws.max_row + 1
    ):

        college_id = ws.cell(
            row=row,
            column=1
        ).value

        if college_id:
            existing_ids[str(college_id)] = row


    for student in students:

        college_id = str(
            student.get("college_id", "")
        )

        if college_id in existing_ids:
            continue

        ws.append([
            student.get("college_id", ""),
            student.get("name", ""),
            student.get("branch", ""),
            student.get("year", ""),
            student.get("section", "")
        ])

    wb.save(EXCEL_FILE)


def save_attendance(
    recognized_students,
    students
):

    try:

        sync_students_to_excel(
            students
        )

        wb = load_workbook(
            EXCEL_FILE
        )

        ws = wb["Attendance"]

        today = date.today().strftime(
            "%d-%m-%Y"
        )

        date_column = None

        for col in range(
            1,
            ws.max_column + 1
        ):

            if ws.cell(
                row=1,
                column=col
            ).value == today:

                date_column = col
                break


        if date_column is None:

            date_column = ws.max_column + 1

            ws.cell(
                row=1,
                column=date_column
            ).value = today


        row_map = {}

        for row in range(
            2,
            ws.max_row + 1
        ):

            college_id = ws.cell(
                row=row,
                column=1
            ).value

            if college_id:

                row_map[
                    str(college_id)
                ] = row


        count = 0

        for student_id in recognized_students:

            student = recognized_students[
                student_id
            ]

            college_id = str(
                student.get(
                    "college_id",
                    ""
                )
            )

            if college_id in row_map:

                ws.cell(
                    row=row_map[college_id],
                    column=date_column
                ).value = "PRESENT"

                count += 1


        wb.save(EXCEL_FILE)

        return True, count

    except PermissionError:

        return False, (
            "Attendance.xlsx is open. "
            "Please close the Excel file and try again."
        )

    except Exception as e:

        return False, str(e)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 College Attendance")

page = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Student Registration",
        "Attendance",
        "Registered Students",
        "Attendance Records"
    ]
)


students = load_students()
embeddings = load_embeddings()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🎓 College Attendance Dashboard")

    today = date.today().strftime(
        "%d-%m-%Y"
    )

    present_count = 0

    if os.path.exists(EXCEL_FILE):

        try:

            wb = load_workbook(
                EXCEL_FILE,
                data_only=True
            )

            ws = wb["Attendance"]

            today_column = None

            for col in range(
                1,
                ws.max_column + 1
            ):

                if ws.cell(
                    row=1,
                    column=col
                ).value == today:

                    today_column = col
                    break

            if today_column:

                for row in range(
                    2,
                    ws.max_row + 1
                ):

                    if ws.cell(
                        row=row,
                        column=today_column
                    ).value == "PRESENT":

                        present_count += 1

        except Exception:
            pass


    total_students = len(students)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Registered Students",
        total_students
    )

    col2.metric(
        "Present Today",
        present_count
    )

    col3.metric(
        "Attendance Date",
        today
    )


    st.divider()

    st.info(
        "Upload a photo from the Attendance section "
        "to mark recognized students as present."
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

elif page == "Student Registration":

    st.title("👨‍🎓 Student Registration")

    st.write(
        "Register a student using 3 to 5 clear face photos."
    )

    with st.form(
        "registration_form"
    ):

        name = st.text_input(
            "Student Name"
        )

        college_id = st.text_input(
            "College ID / Roll Number"
        )

        branch = st.text_input(
            "Branch"
        )

        year = st.selectbox(
            "Year",
            [
                "1st Year",
                "2nd Year",
                "3rd Year",
                "4th Year"
            ]
        )

        section = st.text_input(
            "Section"
        )

        photos = st.file_uploader(
            "Upload 3 to 5 face photos",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            accept_multiple_files=True
        )

        submitted = st.form_submit_button(
            "Register Student"
        )


    if submitted:

        if not name.strip():
            st.error(
                "Please enter the student name."
            )
            st.stop()


        if not college_id.strip():
            st.error(
                "Please enter the College ID."
            )
            st.stop()


        if not photos:

            st.error(
                "Please upload at least 3 photos."
            )
            st.stop()


        if len(photos) < MIN_REGISTRATION_PHOTOS:

            st.error(
                "Please upload at least 3 photos."
            )
            st.stop()


        if len(photos) > MAX_REGISTRATION_PHOTOS:

            st.error(
                "Please upload maximum 5 photos."
            )
            st.stop()


        # Check duplicate ID
        duplicate = any(
            str(s.get("college_id", "")).strip()
            == college_id.strip()
            for s in students
        )


        if duplicate:

            st.error(
                "This College ID is already registered."
            )
            st.stop()


        new_student_id = (
            "student_"
            + college_id.strip()
        )


        student_folder = os.path.join(
            STUDENT_FACES_DIR,
            college_id.strip()
        )

        os.makedirs(
            student_folder,
            exist_ok=True
        )


        new_embeddings = []

        valid_photos = 0

        for index, photo in enumerate(
            photos,
            start=1
        ):

            image = bytes_to_image(
                photo.getvalue()
            )

            if image is None:

                st.warning(
                    f"Photo {index} could not be read."
                )

                continue


            embedding, error = create_embedding(
                image
            )

            if embedding is None:

                st.warning(
                    f"Photo {index} skipped: {error}"
                )

                continue


            new_embeddings.append(
                embedding
            )

            valid_photos += 1


            photo_path = os.path.join(
                student_folder,
                f"photo_{index}.jpg"
            )

            cv2.imwrite(
                photo_path,
                image
            )


        if valid_photos < 3:

            st.error(
                "Registration failed. "
                "At least 3 valid single-person face photos are required."
            )

            st.stop()


        # Save student
        student_data = {
            "student_id": new_student_id,
            "college_id": college_id.strip(),
            "name": name.strip(),
            "college_name": "CPU",
            "branch": branch.strip(),
            "year": year,
            "section": section.strip()
        }


        students.append(
            student_data
        )

        save_students(
            students
        )


        # Save multiple embeddings
        embeddings[new_student_id] = new_embeddings

        save_embeddings(
            embeddings
        )


        # Update Excel
        try:

            sync_students_to_excel(
                students
            )

        except PermissionError:

            st.warning(
                "Student registered, but Excel is open. "
                "Close Attendance.xlsx to update it."
            )


        st.success(
            f"{name} has been registered successfully."
        )

        st.info(
            f"{valid_photos} face photos and embeddings "
            "were added to the recognition database."
        )


# ============================================================
# ATTENDANCE
# ============================================================

elif page == "Attendance":

    st.title("📸 Mark Attendance")

    st.write(
        "Upload a single or group photo. "
        "Only confidently recognized registered students "
        "will be marked present."
    )


    uploaded = st.file_uploader(
        "Upload Attendance Photo",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    camera = st.camera_input(
        "Or take a photo"
    )


    selected_file = (
        uploaded
        if uploaded
        else camera
    )


    if selected_file:

        image = bytes_to_image(
            selected_file.getvalue()
        )

        if image is None:

            st.error(
                "Unable to read this image."
            )
            st.stop()


        faces = detect_faces(
            image
        )


        if len(faces) == 0:

            st.warning(
                "No faces were detected."
            )
            st.stop()


        st.info(
            f"{len(faces)} face(s) detected."
        )


        recognized_students = {}


        for face in faces:

            student_id, score = recognize_face(
                image,
                face,
                embeddings
            )


            if student_id is not None:

                if student_id not in recognized_students:

                    student = next(
                        (
                            s for s in students
                            if s.get("student_id")
                            == student_id
                        ),
                        None
                    )

                    if student:

                        recognized_students[
                            student_id
                        ] = student


        st.divider()


        if recognized_students:

            st.success(
                f"{len(recognized_students)} "
                "student(s) recognized."
            )

            st.subheader(
                "Recognized Students"
            )


            for student in recognized_students.values():

                st.markdown(
                    f"""
                    ### {student.get("name", "")}

                    **College ID:** {student.get("college_id", "")}  
                    **Branch:** {student.get("branch", "")}  
                    **Year:** {student.get("year", "")}  
                    **Section:** {student.get("section", "")}
                    """
                )

                st.divider()


            if st.button(
                "✅ Mark Attendance",
                use_container_width=True
            ):

                success, result = save_attendance(
                    recognized_students,
                    students
                )


                if success:

                    st.success(
                        f"Attendance marked successfully for "
                        f"{result} student(s)."
                    )

                    st.info(
                        "Students who were not recognized "
                        "were left blank and were not marked absent."
                    )

                else:

                    st.error(
                        f"Attendance could not be saved: {result}"
                    )


        else:

            st.warning(
                "No registered student was confidently recognized."
            )


# ============================================================
# REGISTERED STUDENTS
# ============================================================

elif page == "Registered Students":

    st.title("👨‍🎓 Registered Students")

    if not students:

        st.info(
            "No students are registered yet."
        )

    else:

        search = st.text_input(
            "Search by name or College ID"
        ).strip().lower()


        filtered = []

        for student in students:

            name = str(
                student.get("name", "")
            ).lower()

            college_id = str(
                student.get("college_id", "")
            ).lower()


            if (
                not search
                or search in name
                or search in college_id
            ):

                filtered.append(
                    student
                )


        st.write(
            f"Showing {len(filtered)} student(s)."
        )


        for student in filtered:

            with st.container(
                border=True
            ):

                st.subheader(
                    student.get(
                        "name",
                        ""
                    )
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.write(
                    f"**College ID**\n\n"
                    f"{student.get('college_id', '')}"
                )

                c2.write(
                    f"**Branch**\n\n"
                    f"{student.get('branch', '')}"
                )

                c3.write(
                    f"**Year**\n\n"
                    f"{student.get('year', '')}"
                )

                c4.write(
                    f"**Section**\n\n"
                    f"{student.get('section', '')}"
                )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page == "Attendance Records":

    st.title("📊 Attendance Records")


    if not os.path.exists(EXCEL_FILE):

        st.info(
            "No attendance records are available yet."
        )

    else:

        try:

            wb = load_workbook(
                EXCEL_FILE,
                data_only=True
            )

            ws = wb["Attendance"]


            headers = []

            for col in range(
                1,
                ws.max_column + 1
            ):

                headers.append(
                    ws.cell(
                        row=1,
                        column=col
                    ).value
                )


            records = []


            for row in range(
                2,
                ws.max_row + 1
            ):

                record = {}

                for col, header in enumerate(
                    headers,
                    start=1
                ):

                    record[str(header)] = (
                        ws.cell(
                            row=row,
                            column=col
                        ).value
                    )

                records.append(
                    record
                )


            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )


            with open(
                EXCEL_FILE,
                "rb"
            ) as f:

                st.download_button(
                    "⬇️ Download Attendance Excel",
                    data=f,
                    file_name="Attendance.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    )
                )


        except Exception as e:

            st.error(
                "Unable to read attendance records."
            )

            st.code(
                str(e)
            )