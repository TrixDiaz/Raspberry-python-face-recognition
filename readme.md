# Raspberry Pi Face Recognition System

A comprehensive face recognition system designed for Raspberry Pi with motion detection, GPIO control, and real-time video processing capabilities.

## 🚀 Features

- **Real-time Face Recognition**: High-accuracy face detection and recognition using deep learning
- **Motion Detection**: Advanced motion detection with visual indicators
- **GPIO Integration**: Hardware control for authorized access (LEDs, relays, etc.)
- **Multi-person Support**: Train and recognize multiple individuals
- **Live Video Feed**: Real-time camera preview with FPS monitoring
- **Authorization System**: Distinguish between known and authorized users
- **Professional UI**: Clean interface with status indicators and notifications

## 📋 Requirements

### Hardware

- Raspberry Pi 4 (recommended) or Raspberry Pi 3B+
- Raspberry Pi Camera Module v2 or v3
- MicroSD card (32GB+ recommended)
- Optional: LED or relay for GPIO control

### Software

- Raspberry Pi OS (Bullseye or newer)
- Python 3.7+
- OpenCV
- Face Recognition library
- PiCamera2

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Raspberry-python-face-recognition
```

### 2. Install Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv cmake

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Install PiCamera2
sudo apt install -y python3-picamera2

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Enable Camera Interface

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot when prompted
```

## 📖 Usage

### 1. Training the Model

#### Capture Training Images

```bash
python image_capture.py
```

- Press **SPACE** to capture photos
- Press **Q** to quit
- Images are saved in `dataset/[person_name]/` directory

#### Train the Recognition Model

```bash
python model_training.py
```

- Processes all images in the dataset folder
- Creates `encodings.pickle` file with face encodings

### 2. Running Face Recognition

#### Basic Face Recognition (No GPIO)

```bash
python facial_recognition.py
```

#### Face Recognition with GPIO Control

```bash
python facial_recognition_hardware.py
```

### 3. Controls

- **Q**: Quit the application
- **SPACE**: Capture training images (in capture mode)

## ⚙️ Configuration

### Face Recognition Settings

Edit the configuration variables in the Python files:

```python
DISTANCE_THRESHOLD = 0.4  # Recognition sensitivity (0.3-0.6 recommended)
cv_scaler = 4  # Performance scaling factor
```

### Authorization Settings

In `facial_recognition_hardware.py`, modify the authorized users:

```python
authorized_names = ["trix", "john", "jane"]  # Add authorized user names
```

### GPIO Configuration

```python
output = LED(14)  # Change pin number as needed
```

## 📁 Project Structure

```
Raspberry-python-face-recognition/
├── dataset/                    # Training images directory
│   └── [person_name]/         # Individual person folders
├── licenses/                   # Third-party licenses
├── facial_recognition.py      # Basic face recognition
├── facial_recognition_hardware.py  # Face recognition with GPIO
├── image_capture.py           # Training image capture
├── model_training.py          # Model training script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🎯 Features Explained

### Face Recognition

- Uses HOG (Histogram of Oriented Gradients) for face detection
- Employs deep learning models for face encoding
- Supports real-time processing with configurable performance scaling

### Motion Detection

- Background subtraction algorithm for motion detection
- Visual bell icon indicator in upper right corner
- Configurable sensitivity settings

### Authorization System

- **Green Box**: Authorized users
- **Orange Box**: Known but not authorized users
- **Red Box**: Unknown users
- GPIO control for authorized access

### Performance Optimization

- Frame scaling for improved performance
- FPS monitoring and display
- Efficient memory management

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected

```bash
# Check camera connection
vcgencmd get_camera

# Expected output: supported=1 detected=1
```

#### Low FPS Performance

- Increase `cv_scaler` value (e.g., from 4 to 8)
- Reduce camera resolution in configuration
- Ensure adequate power supply (2.5A+ recommended)

#### Recognition Accuracy Issues

- Adjust `DISTANCE_THRESHOLD` (lower = more strict)
- Capture more training images (20-50 per person)
- Ensure good lighting conditions
- Use high-quality images for training

#### GPIO Issues

- Check pin connections
- Verify GPIO pin numbers
- Ensure proper power supply

### Performance Tips

- Use a fast microSD card (Class 10 or better)
- Enable GPU memory split: `sudo raspi-config` → Advanced Options → Memory Split → 128
- Close unnecessary applications
- Use a powered USB hub for additional peripherals

## 📊 System Requirements

| Component | Minimum        | Recommended    |
| --------- | -------------- | -------------- |
| RAM       | 1GB            | 4GB+           |
| Storage   | 8GB            | 32GB+          |
| CPU       | ARM Cortex-A53 | ARM Cortex-A72 |
| Camera    | 5MP            | 8MP+           |
| Power     | 2A             | 3A+            |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Face Recognition Library](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for hardware platform

## 📞 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for educational and personal use. Ensure compliance with local privacy laws and regulations when deploying in public or commercial environments.
