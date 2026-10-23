import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import base64
import io
import numpy as np
from PIL import Image
from deepface import DeepFace

def clean_base64(b64_string: str) -> bytes:
    clean_str = b64_string.strip()
    if "," in clean_str:
        clean_str = clean_str.split(",", 1)[1].strip()
    return base64.b64decode(clean_str)

def decode_image_to_numpy(b64_string: str) -> np.ndarray:
    try:
        image_bytes = clean_base64(b64_string)
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted base64 image data: {str(e)}")

def warmup_model(model_name: str = "Facenet512"):
    print(f"[*] Pre-warming face model: {model_name}...")
    try:
        DeepFace.build_model(model_name=model_name)
        print(f"[OK] {model_name} loaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Could not pre-warm model {model_name}: {e}")

    print("[*] Pre-warming anti-spoofing model: Fasnet...")
    try:
        from deepface.modules import modeling
        modeling.build_model(task="spoofing", model_name="Fasnet")
        print("[OK] Fasnet anti-spoofing loaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Could not pre-warm Fasnet: {e}")

def extract_face_embedding(
    image_base64: str,
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
    enforce_liveness: bool = True,
    strict_single_face: bool = True,
    check_quality: bool = True
) -> dict:
    img_rgb = decode_image_to_numpy(image_base64)

    # 1. Detect faces with strict error messaging and anti-spoofing
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend=detector_backend,
            enforce_detection=True,
            anti_spoofing=enforce_liveness
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "could not be detected" in err_msg or "no face" in err_msg or "facenotdetected" in err_msg:
            raise ValueError("No face detected in camera. Please ensure your face is clearly visible inside the oval.")
        raise ValueError(f"Face detection failed: {str(e)}")

    if not face_objs or len(face_objs) == 0:
        raise ValueError("No face detected in camera. Please ensure your face is clearly visible inside the oval.")

    # 2. Face count and positioning verification
    if strict_single_face and len(face_objs) > 1:
        raise ValueError(
            "Multiple faces detected. Exactly one person must be visible in front of the camera during registration."
        )

    # During daily clock-in/out, colleagues in the office background may be detected.
    # Select the dominant primary face (largest area and closest to the center oval).
    if len(face_objs) > 1:
        img_h, img_w, _ = img_rgb.shape
        center_x, center_y = img_w / 2.0, img_h / 2.0

        def face_prominence_score(item):
            area_info = item.get("facial_area", {})
            w = area_info.get("w", 0)
            h = area_info.get("h", 0)
            area = float(w * h)
            fx = float(area_info.get("x", 0)) + (w / 2.0)
            fy = float(area_info.get("y", 0)) + (h / 2.0)
            dist_from_center = np.sqrt((fx - center_x) ** 2 + (fy - center_y) ** 2)
            return area / (1.0 + (dist_from_center / (img_w * 0.5)))

        face_objs.sort(key=face_prominence_score, reverse=True)

    face_info = face_objs[0]
    area_info = face_info.get("facial_area", {})
    w = area_info.get("w", 0)
    h = area_info.get("h", 0)

    # 3. Framing Quality Checks (Ensures the enrolled baseline vector is high-grade)
    if check_quality:
        # Minimum size requirement: Face must not be tiny or far away
        if w < 90 or h < 90:
            raise ValueError("Face is too far away. Please move closer so your face fills the oval guide.")

        # Centering check: Face center must be well within the camera frame
        img_h, img_w, _ = img_rgb.shape
        center_x, center_y = img_w / 2.0, img_h / 2.0
        fx = float(area_info.get("x", 0)) + (w / 2.0)
        fy = float(area_info.get("y", 0)) + (h / 2.0)
        if abs(fx - center_x) > (img_w * 0.40) or abs(fy - center_y) > (img_h * 0.40):
            raise ValueError("Face is not centered. Please align your face inside the oval guide.")

        # Confidence check: Guard against low light or extreme blur
        confidence = float(face_info.get("confidence", 1.0))
        if confidence < 0.70:
            raise ValueError("Face image is unclear or under low lighting. Please face a light source and look straight at the camera.")

    # 4. Active Liveness Check on the enrolled face
    if enforce_liveness:
        is_real = face_info.get("is_real")
        antispoof_score = face_info.get("antispoof_score", 0.0)
        if is_real is False:
            raise ValueError(
                "Liveness check failed: Spoof or non-live face detected. Please look directly at the camera with live presence."
            )

    # 5. Extract 512-dimension vector embedding directly from the detected face
    try:
        results = DeepFace.represent(
            img_path=face_info["face"],
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False
        )
    except Exception as e:
        raise ValueError(f"Face embedding generation failed: {str(e)}")

    if not results or len(results) == 0:
        raise ValueError("No face embedding generated. Please try again with clear lighting.")

    face_data = results[0]
    embedding = face_data.get("embedding", [])
    confidence = float(face_info.get("confidence", 1.0))

    return {
        "status": "success",
        "model_name": model_name,
        "embedding": embedding,
        "face_confidence": confidence,
        "is_real": face_info.get("is_real", True),
        "antispoof_score": float(face_info.get("antispoof_score", 1.0)) if enforce_liveness else None
    }

def calculate_cosine_distance(vector_a: list, vector_b: list) -> float:
    a = np.array(vector_a, dtype=np.float32)
    b = np.array(vector_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding vector has zero magnitude.")

    dot_product = np.dot(a, b)
    similarity = dot_product / (norm_a * norm_b)
    similarity = float(np.clip(similarity, -1.0, 1.0))
    distance = 1.0 - similarity
    return float(distance)

def verify_face_embedding(
    live_image_base64: str,
    stored_embedding: list,
    model_name: str = "Facenet512",
    detector_backend: str = "opencv",
    threshold: float = 0.28,
    enforce_liveness: bool = True
) -> dict:
    # 1. Extract live embedding with liveness check, allowing office background & tolerance
    extracted = extract_face_embedding(
        image_base64=live_image_base64,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_liveness=enforce_liveness,
        strict_single_face=False,
        check_quality=False
    )
    live_embedding = extracted["embedding"]

    if len(live_embedding) != len(stored_embedding):
        raise ValueError(
            f"Dimension mismatch: Live vector has {len(live_embedding)} dimensions, "
            f"but stored vector has {len(stored_embedding)} dimensions. "
            f"Ensure the same model ({model_name}) was used for enrollment."
        )

    # 2. Compute mathematical cosine distance
    distance = calculate_cosine_distance(live_embedding, stored_embedding)
    matched = distance <= threshold

    return {
        "matched": bool(matched),
        "distance": round(distance, 4),
        "threshold": float(threshold),
        "similarity_metric": "cosine",
        "model_name": model_name,
        "is_liveness_verified": extracted.get("is_real", True)
    }
