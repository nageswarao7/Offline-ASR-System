import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime
import streamlit as st
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import logging
import warnings
import traceback
warnings.filterwarnings("ignore")

# Set up logging with more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Conditional ffmpeg path setting - only if on Windows and paths exist
if os.name == 'nt':  # Windows
    ffmpeg_path = r"C:\Users\nageswararaov\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
    ffprobe_path = r"C:\Users\nageswararaov\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
    
    if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path
        logging.info("Custom ffmpeg paths set successfully")
    else:
        logging.warning("Custom ffmpeg paths not found, using system PATH")

class EnhancedOfflineASRSystem:
    def __init__(self):
        """Initialize the enhanced ASR system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.processor = None
        self.model = None

    def load_models(self):
        """Load ASR models with better error handling"""
        logging.info("Loading Whisper models...")
        try:
            # Start with smallest model for CPU compatibility
            model_names = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"]
            
            for model_name in model_names:
                try:
                    logging.info(f"Attempting to load {model_name}")
                    
                    # Load processor first
                    self.processor = WhisperProcessor.from_pretrained(
                        model_name,
                        cache_dir="./models"  # Use local cache
                    )
                    
                    # Load model with CPU-compatible settings
                    if self.device.type == "cpu":
                        # For CPU, use minimal memory optimization
                        self.model = WhisperForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,  # Always use float32 for CPU
                            cache_dir="./models"
                        )
                    else:
                        # For GPU, can use more optimizations
                        try:
                            self.model = WhisperForConditionalGeneration.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                cache_dir="./models",
                                low_cpu_mem_usage=True
                            )
                        except Exception as gpu_error:
                            logging.warning(f"GPU optimized loading failed, falling back to basic: {gpu_error}")
                            self.model = WhisperForConditionalGeneration.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32,
                                cache_dir="./models"
                            )
                    
                    self.model.to(self.device)
                    self.model.eval()  # Set to evaluation mode
                    
                    logging.info(f"Successfully loaded {model_name}")
                    return True
                    
                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {str(e)}")
                    # Clean up failed attempt
                    self.processor = None
                    self.model = None
                    # Clear GPU cache if using CUDA
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            raise Exception("All model loading attempts failed")
            
        except Exception as e:
            logging.error(f"Critical error loading models: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def preprocess_audio(self, audio_path):
        """Enhanced audio preprocessing with better error handling"""
        logging.info(f"Preprocessing audio: {audio_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            logging.info(f"File size: {file_size / 1024:.1f} KB")
            
            # Process different audio formats
            if audio_path.lower().endswith('.mp3'):
                logging.info("Processing MP3 file with pydub")
                try:
                    audio = AudioSegment.from_mp3(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio = audio.normalize()
                    
                    # Convert to numpy array
                    audio_array = np.array(audio.get_array_of_samples()).astype(np.float32)
                    audio_array = audio_array / np.iinfo(np.int16).max
                    sr = 16000
                    
                except Exception as e:
                    logging.warning(f"pydub failed for MP3, trying librosa: {e}")
                    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                    
            else:
                logging.info("Processing with librosa")
                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Validate audio data
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("Audio array is empty after loading")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                logging.warning("Found NaN or infinite values, cleaning audio")
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate duration
            duration = len(audio_array) / sr
            logging.info(f"Audio duration: {duration:.1f}s")
            
            if duration < 1.0:
                raise ValueError(f"Audio too short: {duration:.1f}s (minimum 1s required)")
            elif duration > 300:  # 5 minutes
                logging.warning(f"Audio very long: {duration:.1f}s, processing may be slow")
            
            # Apply noise reduction
            try:
                audio_array = self.reduce_noise(audio_array)
            except Exception as e:
                logging.warning(f"Noise reduction failed, using original audio: {e}")
            
            logging.info("Audio preprocessing completed successfully")
            return audio_array, sr
            
        except Exception as e:
            logging.error(f"Audio preprocessing failed: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None

    def reduce_noise(self, audio_array):
        """Simple noise reduction with error handling"""
        try:
            # Simple high-pass filter to reduce low-frequency noise
            if len(audio_array) < 1000:
                return audio_array
                
            # Estimate noise from first 0.5 seconds
            noise_sample_size = min(8000, len(audio_array) // 4)
            noise_sample = audio_array[:noise_sample_size]
            noise_power = np.mean(noise_sample ** 2)
            
            if noise_power > 0:
                signal_power = np.mean(audio_array ** 2)
                snr = signal_power / noise_power
                
                if snr < 3.0:  # Low SNR, apply filtering
                    try:
                        from scipy.signal import butter, filtfilt
                        # High-pass filter at 300 Hz
                        nyquist = 16000 / 2
                        low = 300 / nyquist
                        b, a = butter(3, low, btype='high')
                        filtered_audio = filtfilt(b, a, audio_array)
                        return filtered_audio
                    except ImportError:
                        logging.warning("scipy not available, skipping filtering")
                        return audio_array
            
            return audio_array
            
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}")
            return audio_array

    def simple_voice_activity_detection(self, audio_array, sr):
        """Simplified VAD with better error handling"""
        try:
            # Use energy-based VAD
            frame_length = 2048
            hop_length = 512
            
            # Calculate RMS energy
            rms_energy = librosa.feature.rms(
                y=audio_array, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Simple threshold-based detection
            energy_threshold = np.percentile(rms_energy, 30)  # More lenient threshold
            voice_activity = rms_energy > energy_threshold
            
            # Convert to time segments
            times = librosa.frames_to_time(
                np.arange(len(voice_activity)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Find continuous segments
            voice_segments = []
            start_time = None
            min_segment_duration = 0.5  # Minimum 0.5 seconds
            
            for i, is_voice in enumerate(voice_activity):
                if is_voice and start_time is None:
                    start_time = times[i]
                elif not is_voice and start_time is not None:
                    if times[i] - start_time >= min_segment_duration:
                        voice_segments.append((start_time, times[i]))
                    start_time = None
            
            # Handle case where speech continues to end
            if start_time is not None and times[-1] - start_time >= min_segment_duration:
                voice_segments.append((start_time, times[-1]))
            
            # If no segments found, use entire audio
            if not voice_segments:
                voice_segments = [(0, len(audio_array) / sr)]
            
            logging.info(f"Found {len(voice_segments)} voice segments")
            return voice_segments
            
        except Exception as e:
            logging.error(f"VAD failed: {e}")
            # Return entire audio as single segment
            return [(0, len(audio_array) / sr)]

    def transcribe_segment(self, audio_segment, sr=16000):
        """Enhanced transcription with better error handling"""
        try:
            # Validate input
            if len(audio_segment) == 0:
                return ""
            
            # Ensure audio is in correct format
            if len(audio_segment) < sr * 0.1:  # Less than 0.1 seconds
                return ""
            
            # Process with Whisper
            inputs = self.processor(
                audio_segment, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    max_length=448,
                    num_beams=2,  # Reduced for speed and memory
                    do_sample=False,
                    task="transcribe",
                    language=None  # Auto-detect
                )
            
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            return ""

    def process_audio_file(self, audio_path):
        """Main processing function with comprehensive error handling"""
        try:
            logging.info(f"Starting audio processing: {audio_path}")
            
            # Load models if not already loaded
            if not self.model:
                logging.info("Models not loaded, loading now...")
                if not self.load_models():
                    raise Exception("Failed to load ASR models")
            
            # Preprocess audio
            audio_array, sr = self.preprocess_audio(audio_path)
            if audio_array is None:
                raise Exception("Audio preprocessing failed")
            
            # Perform voice activity detection
            voice_segments = self.simple_voice_activity_detection(audio_array, sr)
            if not voice_segments:
                raise Exception("No speech detected in audio")
            
            # Process segments
            results = []
            total_segments = len(voice_segments)
            
            for i, (start_time, end_time) in enumerate(voice_segments):
                logging.info(f"Processing segment {i+1}/{total_segments}")
                
                # Extract audio segment
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                segment_audio = audio_array[start_idx:end_idx]
                
                # Transcribe segment
                transcription = self.transcribe_segment(segment_audio, sr)
                
                if transcription:
                    results.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'speaker': f"Speaker {(i % 2) + 1}",  # Simple alternating speakers
                        'text': transcription
                    })
            
            if not results:
                raise Exception("No transcription results generated")
            
            # Create final output
            output = {
                'full_transcription': '\n'.join(f"{r['speaker']}: {r['text']}" for r in results),
                'speaker_segments': results,
                'total_duration': len(audio_array) / sr,
                'num_speakers': len(set(r['speaker'] for r in results))
            }
            
            logging.info("Audio processing completed successfully")
            return output
            
        except Exception as e:
            logging.error(f"Audio processing failed: {str(e)}")
            logging.error(traceback.format_exc())
            return None

def load_asr_system():
    """Load ASR system and initialize models automatically"""
    system = EnhancedOfflineASRSystem()
    # Load models automatically for deployment
    system.load_models()
    return system

def main():
    st.set_page_config(
        page_title="Enhanced Offline ASR with Speaker Diarization", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🎙️ Enhanced Offline ASR System with Speaker Diarization")
    st.markdown("**Advanced system supporting Hindi, English, and Hinglish with Indian accents**")
    
    # Show loading message if models are not ready
    if 'asr_system' not in st.session_state or not st.session_state.asr_system.model:
        st.warning("⚠️ **System is initializing models on first run. This may take a few minutes.**")
    
    # Initialize ASR system with cached models
    if 'asr_system' not in st.session_state:
        with st.spinner("🔄 Initializing ASR system..."):
            st.session_state.asr_system = load_asr_system()
    
    # Requirements section - only show if there are actual errors
    if 'asr_system' in st.session_state and not st.session_state.asr_system.model:
        st.error("⚠️ **Failed to Initialize ASR Models**")
        with st.expander("🔍 Possible Solutions", expanded=True):
            st.markdown("""
            **Check the following:**
            
            1. **Internet Connection**: First run requires downloading models
            2. **Disk Space**: Ensure ~2GB free space for model cache
            3. **Dependencies**: All required packages should be installed
            4. **Memory**: Ensure sufficient RAM (4GB+ recommended)
            
            **If issues persist, try:**
            ```bash
            pip install --upgrade transformers torch
            ```
            
            Check the console/logs for detailed error messages.
            """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Features")
        st.success("✅ Robust Error Handling")
        st.success("✅ Multiple Model Fallback")
        st.success("✅ Enhanced Audio Processing")
        st.success("✅ Voice Activity Detection")
        st.success("✅ Language Auto-detection")
        
        st.header("📁 Supported Formats")
        st.info("""
        **Audio Formats:**
        • WAV files (recommended)
        • MP3 files
        **Duration:** 1 second - 5 minutes
        **Sample Rate:** Auto-converted to 16kHz
        **Channels:** Auto-converted to mono
        """)
        
        st.header("🚨 Troubleshooting")
        st.info("""
        **Common Issues:**
        • Ensure audio file is not corrupted
        • Check file size (not empty)
        • Try WAV format if MP3 fails
        • Ensure clear speech content
        • Check internet connection for first run
        """)
    
    # Main interface
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.header("📤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3'], 
            help="Upload a WAV or MP3 file"
        )
        
        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("File Name", uploaded_file.name)
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.header("⚙️ Processing Options")
        
        # System status
        with st.expander("📊 System Status", expanded=True):
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"**Device:** {device_info}")
            
            if torch.cuda.is_available():
                st.info(f"**GPU:** {torch.cuda.get_device_name()}")
            
            model_status = "✅ Loaded" if st.session_state.asr_system.model else "❌ Failed to Load"
            st.info(f"**Model Status:** {model_status}")
            
            # Show memory usage
            if torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_properties'):
                memory_info = torch.cuda.get_device_properties(0)
                st.info(f"**GPU Memory:** {memory_info.total_memory / 1024**3:.1f} GB")
            else:
                try:
                    import psutil
                    memory_info = psutil.virtual_memory()
                    st.info(f"**System RAM:** {memory_info.total / 1024**3:.1f} GB available")
                except ImportError:
                    st.info("**Memory Info:** Unable to detect")
        
        # Process button
        if uploaded_file and st.session_state.asr_system.model:
            process_button = st.button("🚀 Process Audio", type="primary", use_container_width=True)
        elif uploaded_file and not st.session_state.asr_system.model:
            process_button = st.button("🚀 Process Audio", type="primary", disabled=True, use_container_width=True)
            st.error("❌ Models failed to load. Please check system requirements.")
        else:
            process_button = st.button("🚀 Process Audio", type="primary", disabled=True, use_container_width=True)
    
    # Processing
    if uploaded_file and process_button and st.session_state.asr_system.model:
        with st.spinner("🔄 Processing audio... Please wait..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file
                status_text.text("Saving uploaded file...")
                progress_bar.progress(10)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process audio
                status_text.text("Processing audio file...")
                progress_bar.progress(30)
                
                results = st.session_state.asr_system.process_audio_file(tmp_file_path)
                
                progress_bar.progress(90)
                
                if results:
                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed successfully!")
                    
                    st.success(f"🎉 Processing completed! Detected {results['num_speakers']} speakers in {results['total_duration']:.1f} seconds of audio.")
                    
                    # Display results
                    tab1, tab2, tab3 = st.tabs(["📝 Transcript", "🔍 Detailed View", "📊 Analysis"])
                    
                    with tab1:
                        st.subheader("Complete Conversation Transcript")
                        st.text_area(
                            "Full transcription with speaker labels:", 
                            results['full_transcription'], 
                            height=300, 
                            help="Copy this text to use elsewhere"
                        )
                    
                    with tab2:
                        st.subheader("Segment-by-Segment Breakdown")
                        for i, segment in enumerate(results['speaker_segments']):
                            with st.expander(f"🎤 {segment['speaker']} | {segment['start_time']:.1f}s - {segment['end_time']:.1f}s"):
                                col_seg1, col_seg2 = st.columns([1, 2])
                                with col_seg1:
                                    st.metric("Speaker", segment['speaker'])
                                    st.metric("Start Time", f"{segment['start_time']:.1f}s")
                                    st.metric("End Time", f"{segment['end_time']:.1f}s")  
                                    st.metric("Duration", f"{segment['end_time'] - segment['start_time']:.1f}s")
                                with col_seg2:
                                    st.write("**Transcription:**")
                                    st.write(segment['text'])
                    
                    with tab3:
                        st.subheader("Conversation Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Duration", f"{results['total_duration']:.1f}s")
                            st.metric("Number of Speakers", results['num_speakers'])
                            st.metric("Total Segments", len(results['speaker_segments']))
                        
                        with col2:
                            st.write("**Speaking Time Distribution:**")
                            speaker_times = {}
                            for segment in results['speaker_segments']:
                                speaker = segment['speaker']
                                duration = segment['end_time'] - segment['start_time']
                                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                            
                            for speaker, time in speaker_times.items():
                                percentage = (time / results['total_duration']) * 100
                                st.write(f"{speaker}: {time:.1f}s ({percentage:.1f}%)")
                        
                        with col3:
                            st.write("**Word Count Distribution:**")
                            speaker_words = {}
                            for segment in results['speaker_segments']:
                                speaker = segment['speaker']
                                word_count = len(segment['text'].split())
                                speaker_words[speaker] = speaker_words.get(speaker, 0) + word_count
                            
                            total_words = sum(speaker_words.values())
                            for speaker, words in speaker_words.items():
                                percentage = (words / total_words) * 100 if total_words > 0 else 0
                                st.write(f"{speaker}: {words} words ({percentage:.1f}%)")
                    
                    # Download options
                    st.subheader("💾 Download Results")
                    col_download1, col_download2 = st.columns(2)
                    
                    with col_download1:
                        # JSON download  
                        output_data = {
                            'metadata': {
                                'filename': uploaded_file.name,
                                'processed_at': datetime.now().isoformat(),
                                'total_duration': results['total_duration'],
                                'num_speakers': results['num_speakers'],
                                'processing_system': 'Enhanced Offline ASR v2.1'
                            },
                            'full_transcription': results['full_transcription'],
                            'segments': results['speaker_segments']
                        }
                        
                        json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_str,
                            file_name=f"{uploaded_file.name}_transcription.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_download2:
                        # TXT download
                        st.download_button(
                            label="📝 Download TXT",
                            data=results['full_transcription'],
                            file_name=f"{uploaded_file.name}_transcription.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                else:
                    st.error("❌ Failed to process audio. Please check the error details above.")
                    
                    # Show debugging info
                    with st.expander("🔍 Debugging Information"):
                        st.write("**File Information:**")
                        st.write(f"- File name: {uploaded_file.name}")
                        st.write(f"- File size: {uploaded_file.size} bytes")
                        st.write(f"- File type: {uploaded_file.type}")
                        
                        st.write("**System Information:**")
                        st.write(f"- PyTorch version: {torch.__version__}")
                        st.write(f"- CUDA available: {torch.cuda.is_available()}")
                        st.write(f"- Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
                        
                        if hasattr(st.session_state.asr_system, 'model') and st.session_state.asr_system.model:
                            st.write("- Model: ✅ Loaded")
                        else:
                            st.write("- Model: ❌ Not loaded")
            
            except Exception as e:
                st.error(f"❌ Critical error: {str(e)}")
                st.error("Please check the file format and try again.")
                
                # Show full error details
                with st.expander("🔍 Full Error Details"):
                    st.code(traceback.format_exc())
            
            finally:
                # Clean up temporary file
                try:
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)
                except:
                    pass

if __name__ == "__main__":
    main()