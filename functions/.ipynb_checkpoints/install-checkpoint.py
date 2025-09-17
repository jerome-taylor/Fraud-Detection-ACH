import importlib.util
import subprocess
import sys

def check_and_install(package_name: str):
    if importlib.util.find_spec(package_name) is None:
        print(f"{package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package_name}: {e}")
    else:
        print(f"{package_name} is already installed.")