import subprocess
import sys
import os

def main():
    print("Starting ExamInsight AI on http://localhost:8080")
    print("Press Ctrl+C to stop.")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8080",
        "--reload"
    ])

if __name__ == "__main__":
    main()
