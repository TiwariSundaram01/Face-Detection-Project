import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from deepface import DeepFace

img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

t0 = time.time()
try:
    DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)
    print(f'retinaface: success in {time.time()-t0:.3f}s')
except Exception as e:
    print(f'retinaface: failed ({e})')
