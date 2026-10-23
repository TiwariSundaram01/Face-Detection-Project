# Propjini Face Verification Microservice (v2.1.0)

High-performance Deep Learning Facial Recognition, Anti-Spoofing, and Biometric Verification microservice.

---

## 🧭 Interactive API Documentation
- **Swagger UI**: [http://127.0.0.1:80/docs](http://127.0.0.1:80/docs)
- **ReDoc**: [http://127.0.0.1:80/redoc](http://127.0.0.1:80/redoc)
- **Interactive Developer Guide**: [http://127.0.0.1:80/guide](http://127.0.0.1:80/guide)
- **Health Check**: [http://127.0.0.1:80/health](http://127.0.0.1:80/health)

---

## 🔐 Security & Authentication
All biometric endpoints (`/api/v1/*`) are secured via the `X-API-KEY` header:
- **Header**: `X-API-KEY: propjini_face_secret_key_2026`

---

## 🎯 Verification Threshold & Calibration Benchmark

| Parameter | Calibrated Value | Technical Description |
|---|---|---|
| **Deep Learning Model** | `Facenet512` | 512-dimensional normalized facial feature embedding network. |
| **Distance Metric** | Cosine Distance | Scale-invariant angle between vectors: $D(u, v) = 1 - \\frac{\\vec{u} \\cdot \\vec{v}}{\\|\\vec{u}\\| \\|\\vec{v}\\|}$. |
| **Match Threshold** | `0.28` | Calibrated threshold. A pair is matched when $\\text{Distance} \\le 0.28$. |
| **Face Detector** | `opencv` | Low-latency facial detector (~80-120ms). Fallbacks: `retinaface`, `mediapipe`. |
| **Anti-Spoofing** | `Fasnet` | Deep learning liveness classifier; rejects screens, photographs, and printed paper cutouts. |
| **Quality Filter** | 90x90px / Centered | Rejects low-resolution, blurred, or heavily off-center frames. |

---

## 🗂️ Microservice Routes

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/health` | No | Uptime diagnostics, active model name, detector backend, threshold. |
| `GET` | `/guide` | No | Interactive visual documentation portal. |
| `POST` | `/api/v1/extract-embedding` | `X-API-KEY` | Extracts 512-dim vector with anti-spoofing and quality checks. |
| `POST` | `/api/v1/verify-embedding` | `X-API-KEY` | Calculates cosine distance between live camera frame and stored vector. |

---

## 💻 cURL Usage Examples

### 1. Extract Biometric Embedding
```bash
curl -X POST "http://127.0.0.1:80/api/v1/extract-embedding" \\
  -H "X-API-KEY: propjini_face_secret_key_2026" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "enforce_liveness": true,
    "strict_single_face": true,
    "check_quality": true
  }'
```

### 2. Verify Live Frame Against Stored Vector
```bash
curl -X POST "http://127.0.0.1:80/api/v1/verify-embedding" \\
  -H "X-API-KEY: propjini_face_secret_key_2026" \\
  -H "Content-Type: application/json" \\
  -d '{
    "live_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "stored_embedding": [0.0124, -0.0482, 0.0911, "... 512 floats ..."],
    "threshold": 0.28,
    "enforce_liveness": true
  }'
```

---

## 🚀 Running the Microservice Locally

```powershell

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

python -m uvicorn app.main:app --host 127.0.0.1 --port 80 --reload
```
