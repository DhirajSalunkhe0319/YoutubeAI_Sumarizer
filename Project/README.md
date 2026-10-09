# 🎥 YouTube AI Summarizer

An open-source, local AI-powered web application that turns any YouTube video into concise, readable research notes using Hugging Face's **`google/flan-t5-base`** model and **Streamlit**.

**Zero Paid APIs. Zero External LLM Keys. 100% Free & Open-Source.**

---

## 🚀 Key Features

- **Universal YouTube Link Support**: Extracts video IDs seamlessly from standard links, `youtu.be` shortened URLs, YouTube Shorts, embed links, and mobile links.
- **Robust Transcript Ingestion**: Powered by `youtube-transcript-api` (supporting both modern 1.x instance API and legacy backends). Automatically prefers manually authored English captions and falls back gracefully to high-quality auto-generated subtitles.
- **Word-Based Chunking (300 Words)**: Subtitles often lack traditional sentence boundaries or punctuation. Our word-level chunking engine ensures clean, token-efficient chunks that never exceed model attention windows.
- **Local AI Inference with FLAN-T5**: Runs `google/flan-t5-base` directly on your machine.
- **Hardware Agnostic**: Automatic GPU acceleration if NVIDIA CUDA is present (`torch.cuda.is_available()`), with smooth, reliable fallback to CPU.
- **Performance Optimized**: Model weights are cached in memory using `@st.cache_resource`, ensuring instantaneous repeat runs without re-downloading or re-loading models.
- **Progress Tracking**: Real-time progress bar and status steps for long videos with dozens of chunks.
- **Export & Inspection**:
  - One-click `.txt` notes download.
  - Per-chunk summarization breakdown expander.
  - Full cleaned transcript viewer.
  - In-app YouTube video player.

---

## 🛠️ Tech Stack

| Component | Library / Model | Description |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Interactive, responsive web dashboard with custom typography and CSS cards |
| **AI Model** | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | Open-source sequence-to-sequence model fine-tuned on instruction tasks |
| **Transformers** | [Hugging Face Transformers](https://github.com/huggingface/transformers) | Model management, tokenization, and beam search generation |
| **Deep Learning** | [PyTorch](https://pytorch.org/) & [Accelerate](https://github.com/huggingface/accelerate) | Tensor computation and inference engine |
| **Data Source** | [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | Captions and subtitle extraction |
| **Tokenizer Engine** | [SentencePiece](https://github.com/google/sentencepiece) | Subword tokenization backend for T5 |

---

## 📂 Project Structure

```text
youtube-summarizer/
│
├── app.py                  # Streamlit web application & user interface
├── requirements.txt        # Python package dependencies
├── README.md               # Complete project documentation
├── .gitignore              # Ignored files (virtualenvs, cache, etc.)
│
├── src/                    # Modular source code package
│   ├── __init__.py         # Package initializer
│   ├── youtube.py          # URL parsing, regex extraction & validation
│   ├── transcript.py       # Subtitle fetching & text cleaning
│   ├── summarizer.py       # Hugging Face FLAN-T5 model loading & inference
│   └── utils.py            # Word-based chunking & note formatting
│
└── assets/                 # Optional static assets
```

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.10+** (Python 3.10 or 3.11 recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/youtube-summarizer.git
cd youtube-summarizer
```

### 2. Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running Locally

Launch the application with Streamlit:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your default browser at:
`http://localhost:8501`

---

## ⚡ Hardware & GPU Acceleration

The application automatically checks for GPU availability:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- **If CUDA GPU is detected**: The model runs with half-precision (`float16`), providing ultra-fast inference and low VRAM usage. Perfect for Google Colab T4, RTX series GPUs, or cloud GPU instances.
- **If only CPU is available**: The model automatically executes on CPU in full precision (`float32`). No manual flags or configurations needed.

---

## 🌐 Deployment Guidelines

### 1. Google Colab (Free T4 GPU)
Because Google Colab provides a free NVIDIA T4 GPU, running this app there is very fast:
1. Open a new Colab notebook with GPU runtime enabled.
2. Clone this repo and install requirements:
   ```bash
   !pip install -r requirements.txt
   !pip install pyngrok
   ```
3. Run Streamlit using ngrok or localtunnel to expose port 8501:
   ```bash
   !streamlit run app.py & npx localtunnel --port 8501
   ```

### 2. Hugging Face Spaces
- Create a new **Streamlit Space** on Hugging Face.
- Push the repository files.
- Select a Space hardware tier with **at least 4 GB RAM** (or a GPU tier).

### 3. Cloud Virtual Machines (AWS EC2 / GCP / DigitalOcean)
- Spin up a VM (e.g. 2+ vCPU, 4GB+ RAM or GPU instance like AWS `g4dn.xlarge`).
- Clone the repository, install dependencies, and run:
  ```bash
  nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
  ```

> [!NOTE]
> Free hosting tiers limited to 512MB - 1GB RAM (like basic render or basic free tiers) may trigger out-of-memory errors when loading `flan-t5-base` (~990MB uncompressed). Ensure your deployment environment has at least 3-4 GB of available system memory.

---

## 🧪 Testing Scenarios

You can verify the core modules with these tests:

| Test Case | Input | Expected Result |
| :--- | :--- | :--- |
| **Standard Video** | `https://www.youtube.com/watch?v=dQw4w9WgXcQ` | Subtitles fetched → chunked → FLAN-T5 notes rendered |
| **Shortened Link** | `https://youtu.be/dQw4w9WgXcQ?si=test` | Clean ID `dQw4w9WgXcQ` extracted and processed |
| **YouTube Shorts** | `https://www.youtube.com/shorts/<ID>` | Shorts URL parsed seamlessly |
| **Invalid URL** | `https://not-youtube.com/watch?v=123` | Clear warning: unrecognized YouTube domain |
| **No Subtitles** | Video with closed captions disabled | Friendly message: `"Subtitles and transcripts are disabled for this video"` |
| **Long Video** | 30+ minute video (e.g. podcast or lecture) | Processed sequentially with step-by-step progress bar |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
