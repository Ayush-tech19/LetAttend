"""Delete local student and attendance data without touching YuNet/SFace models."""
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent

paths = [
    BASE / "data",
    BASE / "database" / "students.json",
    BASE / "model" / "embeddings.json",
    BASE / "Attendance" / "Attendance.xlsx",
    BASE / "student_faces",
]

for path in paths:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

print("All student/attendance data removed.")
print("YuNet/SFace model files were not touched.")
