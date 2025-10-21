# Video Weight Tool

A simple Python app to speed up processing weights from scale videos. The app also allows real-time trimming, cropping, and editing of videos.

## Requirements
- Python 3.8-3.13 (recommended: Python 3.11 or 3.12)
- ffmpeg (system installation required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harrly14/VideoWeightTool
cd VideoWeightTool
```

2. Create a virtual environment (recommended):

This program has several library requirements. To avoid conflicts with system libraries, creating a virtual environment is recommended.

On Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows: 
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg on your system:
- **Windows:** Download from https://ffmpeg.org/ or use `choco install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

5. Verify ffmpeg installation: 
```bash
ffmpeg -version
```

## Usage
```bash
python main.py
```

## Troubleshooting

### NumPy installation fails

If you get compilation errors when installing NumPy, try: 
```bash
pip install --only-binary=numpy -r requirements.txt
```

### Python 3.13+

If you are using a very new Python version, some packages may not have pre-built wheels yet. I recommend using Python 3.11 or 3.12 for best compatibility. You can recreate your virtual environment using one of these Python versions by running the following commands: 

On Linux/macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows: 
```bash
python3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)