import subprocess
import sys
from pathlib import Path

SHUTDOWN_TIMEOUT_SECONDS = 10

def main() -> int:
    root = Path(__file__).resolve().parent
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(root / "User_Interface" / "streamlit_ui.py")],
        cwd=root,
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Ctrl+C is also delivered to the child on a shared console; give it time to exit.
        print("Shutting down Streamlit...")
        try:
            return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.terminate()
            try:
                return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.kill()
                return process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

if __name__ == "__main__":
    sys.exit(main())