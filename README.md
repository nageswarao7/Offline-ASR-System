# 🎙️ Enhanced Offline ASR System with Speaker Diarization

An advanced offline Automatic Speech Recognition (ASR) system that transcribes recorded calls between agents and customers with speaker diarization capabilities. Supports Hindi, English, and Hinglish (Indian accents) speech without requiring any cloud APIs or internet connectivity.

## 🎯 Features

- **Multi-language Support**: Hindi, English, and code-mixed Hinglish
- **Indian Accent Optimization**: Specifically tuned for Indian-accented speech
- **Speaker Diarization**: Identifies and separates different speakers in conversations
- **Offline Processing**: No internet required after initial setup
- **Multiple Audio Formats**: Supports WAV and MP3 files
- **Web Interface**: User-friendly Streamlit interface
- **Robust Error Handling**: Comprehensive error management and fallback mechanisms
- **Export Options**: Download results in JSON or TXT format

## 🏗️ System Architecture

```
Audio Input (WAV/MP3)
        ↓
Audio Preprocessing
    ↓         ↓
Noise Reduction  Format Conversion (16kHz, Mono)
        ↓
Voice Activity Detection (VAD)
        ↓
Audio Segmentation
        ↓
Whisper ASR Processing
        ↓
Speaker Labeling
        ↓
Output Generation
    ↓         ↓
Full Transcript  Speaker-Labeled Segments
```

## 🔧 Technical Components

- **ASR Engine**: OpenAI Whisper (tiny/base/small models)
- **Audio Processing**: librosa, pydub, soundfile
- **Speaker Detection**: Energy-based VAD with clustering
- **Web Interface**: Streamlit
- **Deep Learning**: PyTorch with CUDA support

## 📋 Requirements

### System Requirements
- **RAM**: 4GB+ recommended (8GB+ for optimal performance)
- **Storage**: 2GB free space for model cache
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher

### Hardware Acceleration (Optional)
- **GPU**: CUDA-compatible GPU for faster processing
- **CPU**: Multi-core processor recommended

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nageswarao7/Offline-ASR-System.git
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n asr-env python=3.9
conda activate asr-env

# Or using venv
python -m venv asr-env
# On Windows:
asr-env\Scripts\activate
# On macOS/Linux:
source asr-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


## 🎮 Usage

### 1. Run the Application
```bash
streamlit run app.py
```

### 2. Access the Web Interface
- Open your browser and go to: `http://localhost:8501`
- The system will automatically download and cache models on first run

### 3. Upload and Process Audio
1. Click "Choose an audio file" to upload WAV or MP3 files
2. Supported duration: 1 second to 5 minutes
3. Click "🚀 Process Audio" to start transcription
4. View results in multiple formats
5. Download transcripts as JSON or TXT

### 4. Command Line Usage (Optional)
```python
from your_asr_module import EnhancedOfflineASRSystem

# Initialize system
asr = EnhancedOfflineASRSystem()
asr.load_models()

# Process audio file
results = asr.process_audio_file("path/to/your/audio.wav")

# Print results
print("Full Transcription:")
print(results['full_transcription'])
```

## 📊 Output Format

### Full Transcription
```
Speaker 1: Hello, how can I help you today?
Speaker 2: Hi, I'm calling about my account balance.
Speaker 1: Sure, let me check that for you.
```

### Detailed JSON Output
```json
{
  "metadata": {
    "filename": "call_recording.wav",
    "processed_at": "2024-01-20T10:30:00",
    "total_duration": 45.2,
    "num_speakers": 2
  },
  "full_transcription": "Speaker 1: Hello...",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.2,
      "speaker": "Speaker 1",
      "text": "Hello, how can I help you today?"
    }
  ]
}
```

## 🧪 Testing

### Sample Test Files
The system has been tested with:
- **English calls**: Customer service conversations
- **Hindi calls**: Banking and support calls  
- **Hinglish calls**: Mixed language conversations common in India

### Test Audio Specifications
- **Format**: WAV (recommended) or MP3
- **Duration**: 30 seconds to 3 minutes optimal
- **Quality**: Clear speech, minimal background noise
- **Speakers**: 2-4 speakers maximum for best results

## 🔍 Troubleshooting

### Common Issues and Solutions

**1. Models Not Loading**
```bash
# Clear model cache and retry
rm -rf ./models
# Restart the application
```

**2. Audio Processing Errors**
- Ensure audio file is not corrupted
- Try converting to WAV format
- Check file size (not empty)

**3. Memory Issues**
- Close other applications
- Use smaller Whisper model (tiny instead of base)
- Process shorter audio segments

**4. CUDA/GPU Issues**
```bash
# Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Tips
- **WAV files** process faster than MP3
- **GPU acceleration** significantly improves speed
- **Shorter segments** (1-2 minutes) process more reliably
- **Clear audio** with minimal background noise works best

## 📈 Performance Benchmarks

| Audio Duration | Processing Time (CPU) | Processing Time (GPU) |
|----------------|----------------------|----------------------|
| 30 seconds     | ~15 seconds          | ~5 seconds           |
| 1 minute       | ~30 seconds          | ~10 seconds          |
| 3 minutes      | ~90 seconds          | ~25 seconds          |

*Benchmarks on Intel i7 CPU and NVIDIA GTX 1660 GPU*

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for the excellent ASR models
- **Hugging Face Transformers** for model integration
- **Streamlit** for the web interface framework
- **librosa** for audio processing capabilities

## 📞 Support

For support and questions:
- Create an issue in this repository
- Check the troubleshooting section above
- Review the system requirements

## 🔄 Version History

- **v2.1** - Enhanced error handling and multi-model fallback
- **v2.0** - Added speaker diarization and web interface
- **v1.0** - Basic ASR functionality

---

**Built with ❤️ for offline speech recognition in India**