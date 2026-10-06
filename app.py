import os
import io
import json
import cv2
import numpy as np
import streamlit as st

from datetime import date
from PIL import Image, ImageOps
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")

YUNET_PATH = os.path.join(MODEL_DIR, "yunet.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "sface.onnx")
STUDENTS_FILE = os.path.join(DATABASE_DIR, "students.json")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "embeddings.json")
EXCEL_FILE = os.path.join(ATTENDANCE_DIR, "Attendance.xlsx")


# ============================================================
# SETTINGS
# ============================================================

# Lower than the old detector setting so normal mobile faces are
# less likely to be missed.
DETECTION_SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.30

# SFace cosine-similarity acceptance boundary.
RECOGNITION_THRESHOLD = 0.45

# Extra protection when two registered students get very close scores.
MIN_MATCH_MARGIN = 0.03

MIN_REGISTRATION_PHOTOS = 3
MAX_REGISTRATION_PHOTOS = 5
MAX_IMAGE_SIDE = 1600


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="College Attendance System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# DIRECTORIES + EMPTY DATABASE
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

if not os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4)

if not os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=4)


# ============================================================
# LOAD YUNET + SFACE
# ============================================================

@st.cache_resource
def load_models():
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(f"Missing model file: {YUNET_PATH}")

    if not os.path.exists(SFACE_PATH):
        raise FileNotFoundError(f"Missing model file: {SFACE_PATH}")

    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        DETECTION_SCORE_THRESHOLD,
        NMS_THRESHOLD,
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
# DATABASE HELPERS
# ============================================================

def load_students():
    try:
        with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_students(students):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(students, f, indent=4, ensure_ascii=False)


def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=4)


# ============================================================
# IMAGE HELPERS
# ============================================================

def prepare_image(file_bytes):
    """
    Corrects mobile EXIF orientation, converts to OpenCV BGR,
    and resizes very large photos for stable processing.
    """
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")

        rgb = np.array(pil_image)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]
        largest_side = max(height, width)

        if largest_side > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / largest_side
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        return image
    except Exception:
        return None


def detect_faces(image):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    return [] if faces is None else list(faces)


# ============================================================
# CREATE SFace EMBEDDING FOR REGISTRATION
# ============================================================

def create_embedding(image):
    """Create one embedding from a single-person registration image."""
    faces = detect_faces(image)

    if len(faces) == 0:
        return None, "No face detected. Move closer, face the camera, and try again."

    if len(faces) > 1:
        return None, "Multiple faces detected. Only one person is allowed in a registration photo."

    try:
        aligned = recognizer.alignCrop(image, faces[0])
        feature = recognizer.feature(aligned).astype(np.float32)
        return feature.flatten().tolist(), None
    except Exception as e:
        return None, f"Face feature extraction failed: {e}"


# ============================================================
# RECOGNITION
# ============================================================

def normalize_embedding(value):
    arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 1:
        return [arr.reshape(1, -1)]

    if arr.ndim == 2:
        return [row.reshape(1, -1) for row in arr]

    return []


def recognize_face(image, face, embeddings):
    """
    Compare one detected face against all registered embeddings.
    Returns (student_id or None, best_score, second_best_score).
    """
    try:
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned)
    except Exception:
        return None, 0.0, -1.0

    student_best_scores = []

    for student_id, stored in embeddings.items():
        references = normalize_embedding(stored)
        if not references:
            continue

        best_for_student = -1.0

        for reference in references:
            try:
                score = float(
                    recognizer.match(
                        feature,
                        reference,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )
                )
                best_for_student = max(best_for_student, score)
            except Exception:
                pass

        if best_for_student >= 0:
            student_best_scores.append(
                (student_id, best_for_student)
            )

    if not student_best_scores:
        return None, 0.0, -1.0

    student_best_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    best_student, best_score = student_best_scores[0]
    second_best_score = (
        student_best_scores[1][1]
        if len(student_best_scores) > 1
        else -1.0
    )

    margin_ok = (
        second_best_score < 0
        or best_score - second_best_score >= MIN_MATCH_MARGIN
    )

    if best_score >= RECOGNITION_THRESHOLD and margin_ok:
        return best_student, best_score, second_best_score

    return None, best_score, second_best_score


# ============================================================
# DRAW BOX + NAME
# ============================================================

def draw_face_label(image, face, label, score=None):
    x, y, w, h = [int(v) for v in face[:4]]

    # Green for a recognized student, red for Unknown.
    box_color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        box_color,
        3
    )

    text = label if score is None else f"{label} ({score:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_y = max(y, th + baseline + 10)

    cv2.rectangle(
        image,
        (x, text_y - th - baseline - 8),
        (x + tw + 10, text_y + 4),
        box_color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 5, text_y - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


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

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

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

    wb = load_workbook(EXCEL_FILE)
    ws = wb["Attendance"]
    existing_ids = {}

    for row in range(2, ws.max_row + 1):
        college_id = ws.cell(row=row, column=1).value
        if college_id:
            existing_ids[str(college_id)] = row

    for student in students:
        college_id = str(student.get("college_id", ""))

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


def save_attendance(recognized_students, students):
    try:
        sync_students_to_excel(students)
        wb = load_workbook(EXCEL_FILE)
        ws = wb["Attendance"]

        today = date.today().strftime("%d-%m-%Y")
        date_column = None

        for col in range(1, ws.max_column + 1):
            if ws.cell(row=1, column=col).value == today:
                date_column = col
                break

        if date_column is None:
            date_column = ws.max_column + 1
            ws.cell(row=1, column=date_column).value = today
            ws.cell(row=1, column=date_column).font = Font(bold=True)

        row_map = {}
        for row in range(2, ws.max_row + 1):
            college_id = ws.cell(row=row, column=1).value
            if college_id:
                row_map[str(college_id)] = row

        count = 0
        for student in recognized_students.values():
            college_id = str(student.get("college_id", ""))
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
            "Close the Excel file and try again."
        )
    except Exception as e:
        return False, str(e)


# ============================================================
# APP DATA
# ============================================================

students = load_students()
embeddings = load_embeddings()


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


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("🎓 College Attendance Dashboard")

    today = date.today().strftime("%d-%m-%Y")
    present_count = 0

    if os.path.exists(EXCEL_FILE):
        try:
            wb = load_workbook(EXCEL_FILE, data_only=True)
            ws = wb["Attendance"]
            today_column = None

            for col in range(1, ws.max_column + 1):
                if ws.cell(row=1, column=col).value == today:
                    today_column = col
                    break

            if today_column:
                for row in range(2, ws.max_row + 1):
                    if (
                        ws.cell(row=row, column=today_column).value
                        == "PRESENT"
                    ):
                        present_count += 1
        except Exception:
            pass

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Students", len(students))
    col2.metric("Present Today", present_count)
    col3.metric("Attendance Date", today)

    st.divider()
    st.info(
        "Register students using the camera, then take or upload "
        "a single/group photo in Attendance."
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

elif page == "Student Registration":
    st.title("👨‍🎓 Student Registration")

    st.write(
        "No photo upload is required. Use the camera to capture "
        "3 to 5 photos of the same person from different angles."
    )

    st.info(
        "Recommended: Photo 1 = front, Photo 2 = turn left, "
        "Photo 3 = turn right. Photos 4 and 5 are optional."
    )

    name = st.text_input("Student Name", key="reg_name")
    college_id = st.text_input(
        "College ID / Roll Number",
        key="reg_college_id"
    )
    branch = st.text_input("Branch", key="reg_branch")
    year = st.selectbox(
        "Year",
        ["1st Year", "2nd Year", "3rd Year", "4th Year"],
        key="reg_year"
    )
    section = st.text_input("Section", key="reg_section")

    st.subheader("Capture Face Photos")

    instructions = [
        "Photo 1 — Look straight at the camera",
        "Photo 2 — Turn your face slightly left",
        "Photo 3 — Turn your face slightly right",
        "Photo 4 — Optional: slight upward angle",
        "Photo 5 — Optional: slight downward angle"
    ]

    camera_photos = []
    for i in range(MAX_REGISTRATION_PHOTOS):
        camera_photos.append(
            st.camera_input(
                instructions[i],
                key=f"registration_camera_{i + 1}"
            )
        )

    register_clicked = st.button(
        "✅ Register Student",
        use_container_width=True
    )

    if register_clicked:
        if not name.strip():
            st.error("Please enter the student name.")
            st.stop()

        if not college_id.strip():
            st.error("Please enter the College ID.")
            st.stop()

        duplicate = any(
            str(s.get("college_id", "")).strip()
            == college_id.strip()
            for s in students
        )

        if duplicate:
            st.error("This College ID is already registered.")
            st.stop()

        captured = [p for p in camera_photos if p is not None]

        if len(captured) < MIN_REGISTRATION_PHOTOS:
            st.error(
                "Please capture at least 3 photos using the camera."
            )
            st.stop()

        new_embeddings = []
        valid_photos = 0

        for index, photo in enumerate(captured, start=1):
            image = prepare_image(photo.getvalue())

            if image is None:
                st.warning(
                    f"Photo {index} could not be read. Please retake it."
                )
                continue

            embedding, error = create_embedding(image)

            if embedding is None:
                st.warning(f"Photo {index} skipped: {error}")
                continue

            new_embeddings.append(embedding)
            valid_photos += 1

        if valid_photos < MIN_REGISTRATION_PHOTOS:
            st.error(
                "Registration failed. At least 3 valid single-person "
                "camera photos are required."
            )
            st.info(
                "Make sure only one person is visible, the face is well "
                "lit, and the full face is inside the camera frame."
            )
            st.stop()

        new_student_id = "student_" + college_id.strip()

        student_data = {
            "student_id": new_student_id,
            "college_id": college_id.strip(),
            "name": name.strip(),
            "college_name": "CPU",
            "branch": branch.strip(),
            "year": year,
            "section": section.strip()
        }

        students.append(student_data)
        embeddings[new_student_id] = new_embeddings

        save_students(students)
        save_embeddings(embeddings)

        try:
            sync_students_to_excel(students)
        except PermissionError:
            st.warning(
                "Student was registered, but Attendance.xlsx is open. "
                "Close Excel and try again to sync the sheet."
            )

        st.success(
            f"{name.strip()} has been registered successfully."
        )
        st.info(
            f"{valid_photos} face embeddings were saved. "
            "The registration photos themselves are not stored."
        )
        st.rerun()


# ============================================================
# ATTENDANCE
# ============================================================

elif page == "Attendance":
    st.title("📸 Mark Attendance")

    st.write(
        "Upload or capture one photo. Every detected face is checked "
        "against all registered students."
    )

    uploaded = st.file_uploader(
        "Upload Attendance Photo",
        type=["jpg", "jpeg", "png"],
        key="attendance_upload"
    )

    camera = st.camera_input(
        "Or take an Attendance Photo",
        key="attendance_camera"
    )

    selected_file = uploaded if uploaded is not None else camera

    if selected_file is not None:
        image = prepare_image(selected_file.getvalue())

        if image is None:
            st.error("Unable to read this image.")
            st.stop()

        faces = detect_faces(image)

        if len(faces) == 0:
            st.warning(
                "No faces were detected. Please use a clearer photo "
                "with faces visible and well lit."
            )
            st.stop()

        st.info(f"{len(faces)} face(s) detected.")

        annotated = image.copy()
        recognized_students = {}
        result_rows = []

        for face in faces:
            student_id, score, _ = recognize_face(
                image,
                face,
                embeddings
            )

            if student_id is not None:
                student = next(
                    (
                        s for s in students
                        if s.get("student_id") == student_id
                    ),
                    None
                )

                if student is not None:
                    name_value = student.get("name", "Unknown")
                    recognized_students[student_id] = student

                    draw_face_label(
                        annotated,
                        face,
                        name_value,
                        score
                    )

                    result_rows.append({
                        "Student": name_value,
                        "College ID": student.get("college_id", ""),
                        "Result": "Recognized"
                    })
                    continue

            draw_face_label(annotated, face, "Unknown")
            result_rows.append({
                "Student": "Unknown",
                "College ID": "",
                "Result": "Not Registered"
            })

        st.subheader("Face Recognition Result")

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Detected faces and recognition result",
            use_container_width=True
        )

        st.dataframe(
            result_rows,
            use_container_width=True,
            hide_index=True
        )

        if recognized_students:
            st.success(
                f"{len(recognized_students)} registered student(s) recognized."
            )

            if st.button(
                "✅ Mark Recognized Students Present",
                use_container_width=True,
                key="mark_attendance_button"
            ):
                success, result = save_attendance(
                    recognized_students,
                    students
                )

                if success:
                    st.success(
                        f"Attendance marked successfully for {result} student(s)."
                    )
                    st.info(
                        "Faces that were not recognized were left blank "
                        "and were not marked absent."
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
        st.info("No students are registered yet.")
    else:
        search = st.text_input(
            "Search by name or College ID"
        ).strip().lower()

        filtered = []
        for student in students:
            student_name = str(student.get("name", "")).lower()
            student_id = str(student.get("college_id", "")).lower()

            if (
                not search
                or search in student_name
                or search in student_id
            ):
                filtered.append(student)

        st.write(f"Showing {len(filtered)} student(s).")

        for student in filtered:
            with st.container(border=True):
                st.subheader(student.get("name", ""))

                c1, c2, c3, c4 = st.columns(4)
                c1.write(
                    f"**College ID**\n\n{student.get('college_id', '')}"
                )
                c2.write(
                    f"**Branch**\n\n{student.get('branch', '')}"
                )
                c3.write(
                    f"**Year**\n\n{student.get('year', '')}"
                )
                c4.write(
                    f"**Section**\n\n{student.get('section', '')}"
                )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page == "Attendance Records":
    st.title("📊 Attendance Records")

    if not os.path.exists(EXCEL_FILE):
        st.info("No attendance records are available yet.")
    else:
        try:
            wb = load_workbook(EXCEL_FILE, data_only=True)
            ws = wb["Attendance"]

            headers = [
                ws.cell(row=1, column=col).value
                for col in range(1, ws.max_column + 1)
            ]

            records = []
            for row in range(2, ws.max_row + 1):
                record = {}
                for col, header in enumerate(headers, start=1):
                    record[str(header)] = ws.cell(
                        row=row,
                        column=col
                    ).value
                records.append(record)

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )

            with open(EXCEL_FILE, "rb") as f:
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
            st.error("Unable to read attendance records.")
            st.code(str(e))
