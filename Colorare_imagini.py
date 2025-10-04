#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AI Image Processor - COMPLETELY FIXED VERSION
Problemele rezolvate:
1. send_from_directory syntax fix pentru Flask 2.3.3
2. Aplicarea cumulativa a filtrelor (pe rezultatul anterior)
3. Istoric persistent pe disc
4. Debugging Imbunatatit pentru trasarea problemelor

Rulare:
    pip install -r requirements.txt
    python app.py
"""

import os
import sys
import tempfile

# Detecteaza automat versiunea Python din venv
venv_lib = os.path.join(os.path.dirname(__file__), 'venv', 'lib')
if os.path.exists(venv_lib):
    for item in os.listdir(venv_lib):
        if item.startswith('python'):
            venv_site = os.path.join(venv_lib, item, 'site-packages')
            if os.path.exists(venv_site) and venv_site not in sys.path:
                sys.path.insert(0, venv_site)
                break

# DEBUG - Adauga temporar pentru a vedea ce se Intâmpla
import sys

print(f"DEBUG: Python executable: {sys.executable}", file=sys.stderr)
print(f"DEBUG: sys.path: {sys.path[:3]}", file=sys.stderr)

# Acum Incearca sa importe cv2
try:
    import cv2

    print(f"DEBUG: cv2 importat cu succes din {cv2.__file__}", file=sys.stderr)
except ImportError as e:
    print(f"DEBUG: Eroare la importul cv2: {e}", file=sys.stderr)
    print(f"DEBUG: Verifica daca exista {venv_site}", file=sys.stderr)
    raise

import numpy as np
import requests
import uuid
import json
import time
from flask import Flask, render_template, request, jsonify, session, send_from_directory
import threading
from werkzeug.utils import secure_filename
from flask_session import Session


# Detectare mediu
IS_PRODUCTION = os.environ.get('ENVIRONMENT') == 'production' or \
                os.environ.get('RENDER') is not None or \
                'zooku' in os.environ.get('WEBSITE_HOSTNAME', '').lower()


IS_RENDER = os.environ.get('RENDER') is not None
IS_ZOOKU = 'zooku' in os.environ.get('WEBSITE_HOSTNAME', '').lower() or \
           'zooku' in os.environ.get('VERCEL_URL', '').lower()

# CRITICAL: Pe Zooku folosește directorul curent (fișierele există deja!)
if IS_ZOOKU:
    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'models')
    print(f"Zooku detectat - folosesc modelele existente din: {MODELS_FOLDER}")
elif IS_RENDER:
    MODELS_FOLDER = os.path.join(tempfile.gettempdir(), 'colorization_models')
    print(f"Render detectat - descărcare în: {MODELS_FOLDER}")
else:
    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'models')
    print(f"Local - folosesc: {MODELS_FOLDER}")

os.makedirs(MODELS_FOLDER, exist_ok=True)

# Configurare Flask
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['JSON_AS_ASCII'] = False
Session(app)

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Directoare pentru fisiere
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
TEMPLATES_FOLDER = 'templates'
SESSIONS_FOLDER = 'sessions'
IS_RENDER = os.environ.get('RENDER') == 'true'
print(f"Environment: {'RENDER' if IS_RENDER else 'LOCAL'}")

# Creeaza directoarele necesare
for folder in [UPLOAD_FOLDER, MODELS_FOLDER, TEMPLATES_FOLDER, SESSIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# URL-uri pentru modelele de colorizare
PROTO_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt"
MODEL_URL = "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
PTS_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy"
ALT_MODEL_URL = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/colorization_release_v2.caffemodel"

# Nume fisiere
PROTO = os.path.join(MODELS_FOLDER, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODELS_FOLDER, "colorization_release_v2.caffemodel")
PTS = os.path.join(MODELS_FOLDER, "pts_in_hull.npy")

colorizer_net = None
model_logs = []

model_loading_status = {"status": "loading", "message": "Initializare..."}



def download_file_with_fallback(urls, dst, description="fisier"):
    """Descarca fisier cu multiple URL-uri de backup si progress logging"""
    if os.path.exists(dst):
        file_size = os.path.getsize(dst)
        print(f"{dst} exista deja ({file_size} bytes)")
        # Verifica daca fisierul e valid (mai mare de 1KB)
        if file_size > 1000:
            return dst
        else:
            print(f"Fisier prea mic, sterg si descarc din nou...")
            os.remove(dst)

    if not isinstance(urls, list):
        urls = [urls]

    for i, url in enumerate(urls):
        print(f"Incerc URL {i + 1}/{len(urls)}: {url[:80]}...")
        try:
            # Timeout mai mare pentru Render (conexiune mai lenta)
            timeout = 300 if IS_RENDER else 180

            response = requests.get(url, stream=True, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                start_time = time.time()

                with open(dst, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress la fiecare 10MB
                            if total_size > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                                elapsed = time.time() - start_time
                                speed = downloaded / elapsed / (1024 * 1024)  # MB/s
                                percent = (downloaded / total_size) * 100
                                print(
                                    f" {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({percent:.1f}%) - {speed:.2f}MB/s")

                final_size = os.path.getsize(dst)
                elapsed = time.time() - start_time
                print(f"Descarcare completa: {dst} ({final_size} bytes In {elapsed:.1f}s)")
                return dst
            else:
                print(f"Status code: {response.status_code}")

        except Exception as e:
            print(f"Eroare: {str(e)[:100]}")
            if i < len(urls) - 1:
                print(f"Incerc urmatorul URL...")
            time.sleep(2)  # Pauza Intre Incercari

    raise Exception(f"Nu am putut descarca {description} de la niciun URL")


def log(msg):
    """Adaugă un mesaj în lista de loguri și actualizează status"""
    global model_logs
    model_logs.append(msg)
    print(msg)  # Tot poți păstra print-ul local pentru debug local

def load_models():
    """Încarcă modele și loghează progresul"""
    global colorizer_net, model_loading_status

    try:
        log("=" * 80)
        log("START: Incarcare modele...")
        model_loading_status.update({"status": "loading", "message": "Descarcare modele de colorizare..."})

        # 1. Prototxt
        log("[1/3] Descarcare prototxt...")
        model_loading_status["message"] = "Descarcare prototxt (1/3)..."
        download_file_with_fallback(PROTO_URL, PROTO, "prototxt colorizare")
        log("Prototxt descarcat!")

        # 2. Caffemodel
        log("[2/3] Descarcare model caffemodel (~130MB)...")
        model_loading_status["message"] = "Descarcare model caffemodel (2/3)..."
        download_file_with_fallback([MODEL_URL, ALT_MODEL_URL], MODEL, "model colorizare")
        log("Model caffemodel descarcat!")

        # 3. pts_in_hull
        log("[3/3] Descarcare pts_in_hull.npy...")
        model_loading_status["message"] = "Descarcare pts_in_hull.npy (3/3)..."
        download_file_with_fallback(PTS_URL, PTS, "pts_in_hull.npy")
        log("pts_in_hull descarcat!")

        # 4. Incarcare model
        log("Incarcare model in memorie...")
        model_loading_status["message"] = "Incarcare model in memorie..."
        colorizer_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

        log("Configurare layere...")
        pts_in_hull = np.load(PTS)
        pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        colorizer_net.getLayer(colorizer_net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
        colorizer_net.getLayer(colorizer_net.getLayerId('conv8_313_rh')).blobs = [np.array([2.606], dtype=np.float32)]

        model_loading_status.update({"status": "success", "message": "Model de colorizare incarcat cu succes!"})
        log("MODEL INCARCAT CU SUCCES!")
        log("=" * 80)

    except Exception as e:
        error_msg = f"Eroare la Incarcarea modelelor: {str(e)}"
        model_loading_status.update({"status": "error", "message": error_msg})
        log(error_msg)
        import traceback
        traceback.print_exc()


def load_models_async():
    """Incarca modeleleIn background pentru a nu bloca startup-ul"""
    try:
        print("Incepe incarcarea modelelorIn background...")
        load_models()  # Functia ta deIncarcare
        print("Modele incarcate cu succes!")
    except Exception as e:
        print(f"EROARE laIncarcarea modelelor: {e}")
        import traceback
        traceback.print_exc()
#threading.Thread(target=load_models_async, daemon=True).start()
load_models();
# === FUNCtII PENTRU SESIUNI PERSISTENTE ===
def save_session_to_disk(session_id, session_data):
    """Salveaza sesiunea pe disc In mod atomic si robust"""
    session_file = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
    tmp_file = session_file + ".tmp"

    try:
        # normalizeaza structura minima
        if not isinstance(session_data.get('history', []), list):
            session_data['history'] = []

        with open(tmp_file, 'w', encoding='utf-8', newline='') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # atomic replace (Inlocuieste vechiul fisier doar dupa scriere completa)
        os.replace(tmp_file, session_file)

        print(
            f"DEBUG: Sesiune salvata pentru {session_id} "
            f"cu {len(session_data['history'])} intrari In istoric"
        )
    except Exception as e:
        print(f"Eroare la salvarea sesiunii {session_id}: {e}")


def load_session_from_disk(session_id):
    """Incarca sesiunea de pe disc"""
    session_file = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"DEBUG: Sesiune Incarcata pentru {session_id} cu {len(data.get('history', []))} intrari In istoric")
            return data
        except Exception as e:
            print(f"Eroare la Incarcarea sesiunii: {e}")
    return None


def update_session_data(session_id, new_filepath, operation_name):
    """Actualizeaza sesiunea cu o noua imagine si operatie"""
    print(f"DEBUG: Actualizez sesiunea {session_id} cu operatia '{operation_name}' si fisierul {new_filepath}")

    if session_id not in session:
        # Incarca din disc daca nu e In memorie
        disk_session = load_session_from_disk(session_id)
        if disk_session:
            session[session_id] = disk_session
        else:
            print(f"DEBUG: Nu am gasit sesiunea {session_id}")
            return False

    # Initializeaza history daca nu exista
    if 'history' not in session[session_id]:
        session[session_id]['history'] = []

    # Actualizeaza datele sesiunii
    session[session_id]['filepath'] = os.path.abspath(new_filepath)
    session[session_id]['history'].append({
        'name': operation_name,
        'filepath': os.path.abspath(new_filepath),
        'timestamp': time.time()
    })
    print(f"DEBUG: Istoric actualizat are {len(session[session_id]['history'])} intrari")

    # Salveaza pe disc
    save_session_to_disk(session_id, session[session_id])
    return True


def colorize_image(img):
    """Colorizeaza imaginea folosind modelul AI"""
    if colorizer_net is None:
        raise Exception("Modelul de colorizare nu este Incarcat")

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    h, w = img.shape[:2]
    l_rs = cv2.resize(l, (224, 224))
    l_rs -= 50

    blob = cv2.dnn.blobFromImage(l_rs)
    colorizer_net.setInput(blob)
    ab_dec = colorizer_net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_dec_us = cv2.resize(ab_dec, (w, h))

    lab_out = np.zeros((h, w, 3), dtype=np.float32)
    lab_out[:, :, 0] = l
    lab_out[:, :, 1:] = ab_dec_us

    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    bgr_out = np.clip(bgr_out, 0, 1)
    bgr_out = (255 * bgr_out).astype('uint8')

    return bgr_out


def detect_objects_simple(img):
    """Detectare obiecte simplificata folosind contururi si caracteristici"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:  # Filtru pentru obiecte mici
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(area / 10000, 0.9)  # Pseudo-confidence

            # Clasificare simpla bazata pe forma
            aspect_ratio = w / h
            if aspect_ratio > 1.5:
                obj_class = "dreptunghi/vehicul"
            elif 0.8 < aspect_ratio < 1.2:
                obj_class = "patrat/obiect"
            else:
                obj_class = "obiect vertical"

            objects.append({
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x, y, w, h]
            })

    return objects[:10]  # Limiteaza la 10 obiecte


active_rgb_filters = set()


def apply_filter(img, filter_type, original_img=None, **kwargs):
    """Aplica filtre pe imagine cu sistem de toggle pentru RGB"""
    print(f"DEBUG: Aplicarea filtrului {filter_type} pe imagine cu dimensiunea {img.shape}")

    global active_rgb_filters
    print(f"DEBUG: Filtre RGB active: {active_rgb_filters}")

    try:
        # --- SISTEM TOGGLE PENTRU RGB (se aplica pe imaginea originala) ---
        if filter_type in ["red_channel", "green_channel", "blue_channel"]:
            # Toggle: activeaza/dezactiveaza filtrul
            if filter_type in active_rgb_filters:
                active_rgb_filters.remove(filter_type)
                print(f"DEBUG: Filtrul {filter_type} DEZACTIVAT")
            else:
                active_rgb_filters.add(filter_type)
                print(f"DEBUG: Filtrul {filter_type} ACTIVAT")

            print(f"DEBUG: Filtre RGB active: {active_rgb_filters}")

            # IMPORTANT: Aplicam toate filtrele RGB active pe imaginea ORIGINALa
            if original_img is not None:
                return apply_rgb_filters(original_img, active_rgb_filters)
            else:
                # Fallback - daca nu avem imaginea originala, folosim cea curenta
                return apply_rgb_filters(img, active_rgb_filters)

        # --- RESTUL FILTRELOR (se aplica pe imaginea curenta) ---
        elif filter_type == "grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif filter_type == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            return cv2.transform(img, kernel)

        elif filter_type == "cartoon":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

            data = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_img = segmented_data.reshape(img.shape)

            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.bitwise_and(segmented_img, edges)


        elif filter_type == "rotate":
            angle = kwargs.get("angle", 0)
            print(f"DEBUG: Rotire unghi {angle}°")

            if angle == 0:
                return img

            # Rotiri exacte de 90, 180, 270 fara decupaje
            if angle % 360 == 90 or angle % 360 == -270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle % 360 == -90 or angle % 360 == 270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif abs(angle) % 360 == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            else:
                # Rotire libera cu warpAffine
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Ajustam dimensiunea pentru a evita decuparea
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))

                # Ajustam matricea pentru a centra imaginea
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]

                rotated = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated
        elif filter_type == "blur":
            strength = kwargs.get('strength', 15)
            if strength % 2 == 0:
                strength += 1
            return cv2.GaussianBlur(img, (strength, strength), 0)

        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(img, -1, kernel)

        elif filter_type == "emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            return cv2.filter2D(img, -1, kernel)

        elif filter_type == "edge_detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif filter_type == "vintage":
            result = img.copy().astype(np.float32)
            result[:, :, 0] *= 0.9
            result[:, :, 1] *= 1.1
            result[:, :, 2] *= 1.2
            result = np.clip(result, 0, 255).astype(np.uint8)

            h, w = result.shape[:2]
            kernel = cv2.getGaussianKernel(h, h // 4)
            kernel = np.outer(kernel, kernel)
            kernel = kernel / kernel.max()
            kernel = np.dstack([kernel] * 3)
            return (result * kernel).astype(np.uint8)

        elif filter_type == "negative":
            return 255 - img

        elif filter_type == "brightness":
            # Parametru intensitate: poate fi -100..+100 sau 0..200%, depinde de preferinte
            intensity = kwargs.get('intensity', 0)  # de exemplu 0 = fara modificare

            # Convertim la float pentru calcule
            result = img.astype(np.float32)

            # Ajustam luminozitatea prin adunare
            result += intensity

            # Limitam valorile Intre 0 si 255
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result

        else:
            return img

    except Exception as e:
        print(f"Eroare la aplicarea filtrului {filter_type}: {e}")
        return img


def get_last_non_rgb_image_from_history(session_id):
    """Extrage ultima imagine din istoric care nu este un filtru RGB"""
    if session_id not in session:
        return None

    history = session[session_id].get('history', [])
    if not history:
        return None

    # Cauta de la sfârsitul istoricului catre Inceput
    for item in reversed(history):
        # Verifica daca numele operatiei nu contine filtre RGB
        name_lower = item['name'].lower()
        if not any(rgb_filter in name_lower for rgb_filter in
                   ['canal rosu', 'canal verde', 'canal albastru', 'red channel', 'green channel', 'blue channel']):
            return item['filepath']

    # Daca toate sunt RGB, returneaza originalul
    return session[session_id].get('original_filepath')


@app.route("/test_model")
def test_model():
    try:
        net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route("/test_model_detailed")
def test_model_detailed():
    import time
    results = []

    try:
        results.append({"step": "1. Check files", "status": "start"})
        results.append({
            "proto_exists": os.path.exists(PROTO),
            "model_exists": os.path.exists(MODEL),
            "pts_exists": os.path.exists(PTS),
            "model_size_mb": os.path.getsize(MODEL) / (1024 * 1024) if os.path.exists(MODEL) else 0
        })

        results.append({"step": "2. Load prototxt only", "status": "start"})
        start = time.time()
        net = cv2.dnn.readNetFromCaffe(PROTO)
        results.append({"step": "2. Load prototxt only", "status": "success", "time": time.time() - start})

        results.append({"step": "3. Load full model", "status": "start"})
        start = time.time()
        net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
        results.append({"step": "3. Load full model", "status": "success", "time": time.time() - start})

        results.append({"step": "4. Load pts_in_hull", "status": "start"})
        start = time.time()
        pts_in_hull = np.load(PTS)
        results.append({"step": "4. Load pts_in_hull", "status": "success", "time": time.time() - start})

        results.append({"step": "5. Configure class8_ab", "status": "start"})
        start = time.time()
        pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
        results.append({"step": "5. Configure class8_ab", "status": "success", "time": time.time() - start})

        results.append({"step": "6. Configure conv8_313_rh", "status": "start"})
        start = time.time()
        net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.array([2.606], dtype=np.float32)]
        results.append({"step": "6. Configure conv8_313_rh", "status": "success", "time": time.time() - start})

        return {"status": "success", "results": results}

    except Exception as e:
        results.append({"step": "ERROR", "message": str(e)})
        import traceback
        return {"status": "error", "results": results, "traceback": traceback.format_exc()}


@app.route("/check_files")
def check_files():
    files_info = {}
    for root, dirs, files in os.walk("."):
        for f in files:
            path = os.path.join(root, f)
            files_info[path] = os.path.getsize(path)
    return jsonify(files_info)

@app.route('/api/debug-models')
def debug_models():
    """Debug endpoint pentru modele"""
    return jsonify({
        "models_folder": MODELS_FOLDER,
        "is_zooku": IS_ZOOKU,
        "files": {
            "prototxt": {
                "path": PROTO,
                "exists": os.path.exists(PROTO),
                "size": os.path.getsize(PROTO) if os.path.exists(PROTO) else 0
            },
            "caffemodel": {
                "path": MODEL,
                "exists": os.path.exists(MODEL),
                "size": os.path.getsize(MODEL) if os.path.exists(MODEL) else 0
            },
            "pts_in_hull": {
                "path": PTS,
                "exists": os.path.exists(PTS),
                "size": os.path.getsize(PTS) if os.path.exists(PTS) else 0
            }
        },
        "loading_status": model_loading_status
    })

@app.route("/api/model-logs")
def get_model_logs():
    return jsonify({
        "status": model_loading_status,
        "logs": model_logs
    })

LOG_FILE = "debug_log.jsonl"

@app.route("/debug-save", methods=["GET", "POST"])
def debug_save():
    if request.method == "POST":
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data sent"}), 400

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return jsonify({"success": True})

    # GET doar pentru verificare/log vizual în browser
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.read()
        return f"<pre>{logs}</pre>"
    except FileNotFoundError:
        return "Nu exista log-uri."


@app.route('/api/filter', methods=['POST'])
def apply_image_filter():
    """Aplica un filtru pe imaginea CURENTa si salveaza rezultatul"""
    data = request.json
    session_id = data.get('session_id')
    filter_type = data.get('filter_type')
    filter_params = data.get('params', {})
    # print(f"DEBUG: Aplicare filtru {filter_type} pentru sesiunea {session_id}")

    # Incarca sesiunea daca nu exista In memorie
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # Pentru filtrele RGB, folosim ultima imagine non-RGB din istoric
        if filter_type in ["red_channel", "green_channel", "blue_channel"]:
            base_image_path = get_last_non_rgb_image_from_history(session_id)
            if not base_image_path or not os.path.exists(base_image_path):
                return jsonify({'error': 'Nu s-a gasit imaginea de baza pentru filtrele RGB'}), 400

            # print(f"DEBUG: Aplicare filtru RGB pe ultima imagine non-RGB: {base_image_path}")
            img = cv2.imread(base_image_path)
            original_img = img.copy()  # Pentru parametrul original_img
        else:
            # Pentru alte filtre, folosim imaginea curenta
            current_filepath = session[session_id]['filepath']
            # print(f"DEBUG: Aplicare filtru normal pe imaginea curenta: {current_filepath}")

            if not os.path.exists(current_filepath):
                return jsonify({'error': 'Fisierul curent nu exista'}), 400

            img = cv2.imread(current_filepath)
            original_img = None  # Nu e necesar pentru filtrele normale

        if img is None:
            return jsonify({'error': 'Imaginea nu poate fi citita'}), 400

        # Aplica filtrul cu imaginea de baza daca e necesar
        filtered = apply_filter(img, filter_type, original_img=original_img, **filter_params)

        # Salveaza imaginea filtrata
        timestamp = int(time.time())
        filtered_filename = f"{session_id}_{filter_type}_{timestamp}.png"
        filtered_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filtered_filename))
        cv2.imwrite(filtered_path, filtered)
        # print(f"DEBUG: Imagine filtrata salvata: {filtered_path}")

        # Actualizeaza sesiunea + istoricul
        filter_name = filter_type.replace('_', ' ').title()
        update_session_data(session_id, filtered_path, f'Filtru {filter_name}')
        session[session_id]['filepath'] = filtered_path

        return jsonify({
            'success': True,
            'image': f"/uploads/{filtered_filename}",
            'width': filtered.shape[1],
            'height': filtered.shape[0]
        })

    except Exception as e:
        print(f"DEBUG: Eroare la aplicarea filtrului: {e}")
        return jsonify({'error': f'Eroare la aplicarea filtrului: {str(e)}'}), 500


# Functie helper pentru resetarea filtrelor RGB
@app.route('/api/rgb/reset', methods=['POST'])
def reset_rgb_filters_api():
    """Reseteaza toate filtrele RGB si revine la imaginea originala"""
    data = request.json
    session_id = data.get('session_id')

    if session_id not in session:
        disk_session = load_session_from_disk(session_id)
        if disk_session:
            session[session_id] = disk_session
        else:
            return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # Reseteaza filtrele RGB
        reset_rgb_filters()

        # Revine la imaginea originala
        original_filepath = session[session_id].get('original_filepath')
        if original_filepath and os.path.exists(original_filepath):
            session[session_id]['filepath'] = original_filepath
            save_session_to_disk(session_id, session[session_id])

            img = cv2.imread(original_filepath)
            return jsonify({
                'success': True,
                'image': f"/uploads/{os.path.basename(original_filepath)}",
                'width': img.shape[1],
                'height': img.shape[0],
                'message': 'Filtre RGB resetate'
            })
        else:
            return jsonify({'error': 'Imaginea originala nu exista'}), 400

    except Exception as e:
        return jsonify({'error': f'Eroare la resetarea filtrelor RGB: {str(e)}'}), 500


# Functie pentru a obtine statusul filtrelor RGB
@app.route('/api/rgb/status', methods=['POST'])
def get_rgb_status_api():
    """Obtine statusul curent al filtrelor RGB"""
    return jsonify({
        'success': True,
        'status': get_rgb_status()
    })


def apply_rgb_filters(img, active_filters):
    """Aplica filtrele RGB active pe imaginea originala"""
    if not active_filters:
        return img  # Daca nu sunt filtre active, returneaza imaginea originala

    result = np.zeros_like(img)  # Incepem cu negru

    # Adunam canalele active din imaginea originala
    if "red_channel" in active_filters:
        result[:, :, 2] = img[:, :, 2]  # BGR: Red este pe pozitia 2

    if "green_channel" in active_filters:
        result[:, :, 1] = img[:, :, 1]  # BGR: Green este pe pozitia 1

    if "blue_channel" in active_filters:
        result[:, :, 0] = img[:, :, 0]  # BGR: Blue este pe pozitia 0

    return result


def get_rgb_status():
    """Returneaza starea curenta a filtrelor RGB"""
    return {
        'red_active': 'red_channel' in active_rgb_filters,
        'green_active': 'green_channel' in active_rgb_filters,
        'blue_active': 'blue_channel' in active_rgb_filters,
        'active_filters': list(active_rgb_filters),
        'count': len(active_rgb_filters)
    }


def reset_rgb_filters():
    """Reseteaza toate filtrele RGB"""
    global active_rgb_filters
    active_rgb_filters.clear()
    print("DEBUG: Toate filtrele RGB au fost resetate")


def apply_current_rgb_filters(img):
    """Aplica filtrele RGB curente pe o imagine noua (util pentru refresh)"""
    return apply_rgb_filters(img, active_rgb_filters)


def upscale_image(img, scale_factor=2):
    """Mareste imaginea cu factorul specificat"""
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Returneaza statusul Incarcarii modelelor"""
    return jsonify(model_loading_status)


@app.route('/uploads/<filename>')
def serve_image(filename):
    """Serveste imaginile din directorul uploads - FIXED pentru Flask 2.3.3"""
    try:
        print(f"DEBUG: Servire imagine: {filename}")
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)
    except Exception as e:
        print(f"Eroare la servirea imaginii {filename}: {e}")
        return jsonify({'error': 'Imaginea nu a fost gasita'}), 404


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Incarca o imagine si o salveaza pe disc"""
    if 'image' not in request.files:
        return jsonify({'error': 'Nu s-a gasit fisierul'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nu s-a selectat fisierul'}), 400

    try:
        # Creeaza un ID unic pentru fisier
        session_id = str(uuid.uuid4())
        filename = secure_filename(f"{session_id}_original.png")
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Salveaza fisierul pe disc
        file.save(filepath)
        print(f"DEBUG: Fisier salvat: {filepath}")

        # Citeste imaginea cu OpenCV pentru a obtine dimensiunile
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Imaginea nu poate fi citita'}), 400

        height, width = img.shape[:2]

        # Initializeaza sesiunea cu istoricul
        session_data = {
            'filepath': filepath,  # imagine curenta
            'original_filepath': filepath,  # imagine originala
            'history': [{
                'name': 'Original',
                'filepath': filepath,
                'timestamp': time.time()
            }]
        }

        session[session_id] = session_data

        # Salveaza sesiunea pe disc
        save_session_to_disk(session_id, session_data)
        print(f"DEBUG: Sesiune initializata pentru {session_id}")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'image': f"/uploads/{filename}",
            'width': width,
            'height': height
        })

    except Exception as e:
        print(f"DEBUG: Eroare la upload: {e}")
        return jsonify({'error': f'Eroare la procesare: {str(e)}'}), 500


@app.route('/api/colorize', methods=['POST'])
def colorize():
    """Colorizeaza imaginea CURENTa si salveaza rezultatul"""
    data = request.json
    session_id = data.get('session_id')
    print(f"DEBUG: Colorize pentru sesiunea {session_id}")

    # Incarca sesiunea daca nu exista In memorie
    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    if colorizer_net is None:
        return jsonify({'error': 'Modelul de colorizare nu este Incarcat'}), 500

    try:
        # Foloseste ultima imagine procesata din sesiune
        current_filepath = session[session_id]['filepath']
        print(f"DEBUG: Colorizez imaginea curenta: {current_filepath}")

        if not os.path.exists(current_filepath):
            return jsonify({'error': 'Fisierul curent nu exista'}), 400

        img = cv2.imread(current_filepath)
        if img is None:
            return jsonify({'error': 'Imaginea nu poate fi citita'}), 400

        # Aplica colorizarea AI
        colorized = colorize_image(img)

        # Salveaza imaginea colorizata
        timestamp = int(time.time())
        colorized_filename = f"{session_id}_colorized_{timestamp}.png"
        colorized_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, colorized_filename))
        cv2.imwrite(colorized_path, colorized)
        print(f"DEBUG: Imagine colorizata salvata: {colorized_path}")

        # Actualizeaza sesiunea + istoricul
        update_session_data(session_id, colorized_path, 'Colorare AI')
        session[session_id]['filepath'] = colorized_path
        print(session[session_id]['filepath'])
        return jsonify({
            'success': True,
            'image': f"/uploads/{colorized_filename}",
            'width': colorized.shape[1],
            'height': colorized.shape[0]
        })

    except Exception as e:
        print(f"DEBUG: Eroare la colorizare: {e}")
        return jsonify({'error': f'Eroare la colorizare: {str(e)}'}), 500


@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """Detecteaza obiectele din imaginea CURENTa"""
    data = request.json
    session_id = data.get('session_id')

    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # Foloseste imaginea curenta
        current_filepath = session[session_id]['filepath']
        if not os.path.exists(current_filepath):
            return jsonify({'error': 'Fisierul curent nu exista'}), 400

        img = cv2.imread(current_filepath)
        if img is None:
            return jsonify({'error': 'Imaginea nu poate fi citita'}), 400

        objects = detect_objects_simple(img)

        # Deseneaza bounding boxes pe imaginea curenta
        result_img = img.copy()
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            cv2.putText(result_img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Salveaza rezultatul
        timestamp = int(time.time())
        detected_filename = f"{session_id}_detected_{timestamp}.png"
        detected_path = os.path.join(UPLOAD_FOLDER, detected_filename)
        cv2.imwrite(detected_path, result_img)

        # Actualizeaza sesiunea
        update_session_data(session_id, detected_path, f'Detectare {len(objects)} obiecte')

        return jsonify({
            'success': True,
            'image': f"/uploads/{detected_filename}",
            'objects': objects
        })

    except Exception as e:
        return jsonify({'error': f'Eroare la detectare: {str(e)}'}), 500


@app.route('/api/upscale', methods=['POST'])
def upscale():
    """Mareste rezolutia imaginii CURENTE"""
    data = request.json
    session_id = data.get('session_id')
    scale_factor = float(data.get('scale_factor', 2.0))

    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # Foloseste imaginea curenta
        current_filepath = session[session_id]['filepath']
        if not os.path.exists(current_filepath):
            return jsonify({'error': 'Fisierul curent nu exista'}), 400

        img = cv2.imread(current_filepath)
        if img is None:
            return jsonify({'error': 'Imaginea nu poate fi citita'}), 400

        upscaled = upscale_image(img, scale_factor)

        # Salveaza rezultatul
        timestamp = int(time.time())
        upscaled_filename = f"{session_id}_upscaled_{scale_factor}x_{timestamp}.png"
        upscaled_path = os.path.join(UPLOAD_FOLDER, upscaled_filename)
        cv2.imwrite(upscaled_path, upscaled)

        # Actualizeaza sesiunea
        update_session_data(session_id, upscaled_path, f'Upscale {scale_factor}x')

        return jsonify({
            'success': True,
            'image': f"/uploads/{upscaled_filename}",
            'width': upscaled.shape[1],
            'height': upscaled.shape[0]
        })

    except Exception as e:
        return jsonify({'error': f'Eroare la upscale: {str(e)}'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_image():
    """Reseteaza la imaginea originala"""
    data = request.json
    session_id = data.get('session_id')

    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # Obtine calea imaginii originale
        original_filepath = session[session_id].get('original_filepath')
        if not original_filepath or not os.path.exists(original_filepath):
            return jsonify({'error': 'Imaginea originala nu exista'}), 400

        # Reseteaza imaginea curenta la original
        session[session_id]['filepath'] = original_filepath
        session[session_id]['history'] = [{
            'name': 'Original',
            'filepath': original_filepath,
            'timestamp': time.time()
        }]

        # Salveaza sesiunea actualizata
        save_session_to_disk(session_id, session[session_id])

        # Citeste imaginea pentru dimensiuni
        img = cv2.imread(original_filepath)
        height, width = img.shape[:2]

        return jsonify({
            'success': True,
            'image': f"/uploads/{os.path.basename(original_filepath)}",
            'width': width,
            'height': height
        })

    except Exception as e:
        return jsonify({'error': f'Eroare la resetare: {str(e)}'}), 500


@app.route('/api/history', methods=['POST'])
def get_history():
    """Obtine istoricul imaginilor din sesiune"""
    data = request.json
    session_id = data.get('session_id')
    print(f"DEBUG: Cerere istoric pentru sesiunea {session_id}")

    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        history = session[session_id].get('history', [])
        print(f"DEBUG: Istoric gasit cu {len(history)} intrari")

        # Adauga URL-uri pentru frontend si verifica existenta fisierelor
        formatted_history = []
        for item in history:
            if os.path.exists(item['filepath']):
                formatted_history.append({
                    'name': item['name'],
                    'url': f"/uploads/{os.path.basename(item['filepath'])}",
                    'timestamp': item.get('timestamp', 0)
                })

        print(f"DEBUG: Istoric formatat cu {len(formatted_history)} intrari valide")

        return jsonify({
            'success': True,
            'history': formatted_history
        })

    except Exception as e:
        print(f"DEBUG: Eroare la obtinerea istoricului: {e}")
        return jsonify({'error': f'Eroare la obtinerea istoricului: {str(e)}'}), 500


@app.route('/api/history/goto', methods=['POST'])
def goto_history():
    """Mergi la o imagine din istoric"""
    data = request.json
    session_id = data.get('session_id')
    index = int(data.get('index', 0))

    # if session_id not in session:
    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        history = session[session_id].get('history', [])
        if 0 <= index < len(history):
            selected_filepath = history[index]['filepath']

            if os.path.exists(selected_filepath):
                # Actualizeaza imaginea curenta
                # session[session_id]['filepath'] = selected_filepath
                save_session_to_disk(session_id, session[session_id])

                # Citeste imaginea pentru dimensiuni
                img = cv2.imread(selected_filepath)
                height, width = img.shape[:2]

                return jsonify({
                    'success': True,
                    'image': f"/uploads/{os.path.basename(selected_filepath)}",
                    'width': width,
                    'height': height
                })
            else:
                return jsonify({'error': 'Fisierul din istoric nu mai exista'}), 400
        else:
            return jsonify({'error': 'Index invalid'}), 400

    except Exception as e:
        return jsonify({'error': f'Eroare la navigarea In istoric: {str(e)}'}), 500


@app.route('/api/history/undo', methods=['POST'])
def undo_history():
    data = request.json
    session_id = data.get('session_id')
    session_data = load_session_from_disk(session_id)
    session[session_id] = session_data
    history = session[session_id]['history']
    redo_stack = session[session_id].get('redo', [])

    if len(history) <= 1:
        return jsonify({'error': 'Nu exista istoric pentru undo'}), 400

    last_state = history.pop()
    redo_stack.append(last_state)
    session[session_id]['redo'] = redo_stack
    session[session_id]['history'] = history
    session[session_id]['filepath'] = history[-1]['filepath']
    save_session_to_disk(session_id, session[session_id])

    return jsonify({
        'success': True,
        'filepath': history[-1]['filepath'],
        'image': f"/uploads/{os.path.basename(history[-1]['filepath'])}"
    })


@app.route('/api/history/redo', methods=['POST'])
def redo_history():
    data = request.json
    session_id = data.get('session_id')
    session_data = load_session_from_disk(session_id)
    session[session_id] = session_data
    history = session[session_id]['history']
    redo_stack = session[session_id].get('redo', [])

    if not redo_stack:
        return jsonify({'error': 'Nu exista istoric pentru redo'}), 400

    state = redo_stack.pop()
    history.append(state)
    session[session_id]['redo'] = redo_stack
    session[session_id]['history'] = history
    session[session_id]['filepath'] = state['filepath']
    save_session_to_disk(session_id, session[session_id])

    return jsonify({
        'success': True,
        'filepath': history[-1]['filepath'],
        'image': f"/uploads/{os.path.basename(history[-1]['filepath'])}"
    })


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """sterge istoricul si reseteaza la imaginea originala"""
    data = request.json
    session_id = data.get('session_id')

    disk_session = load_session_from_disk(session_id)
    if disk_session:
        session[session_id] = disk_session
    else:
        return jsonify({'error': 'Sesiune invalida'}), 400

    try:
        # sterge fisierele procesate (pastreaza doar originalul)
        history = session[session_id].get('history', [])
        original_filepath = session[session_id].get('original_filepath')

        for item in history:
            if item['filepath'] != original_filepath and os.path.exists(item['filepath']):
                try:
                    os.remove(item['filepath'])
                except Exception as e:
                    print(f"Nu am putut sterge fisierul {item['filepath']}: {e}")

        # Reseteaza la original
        session[session_id]['filepath'] = original_filepath
        session[session_id]['history'] = [{
            'name': 'Original',
            'filepath': original_filepath,
            'timestamp': time.time()
        }]

        # Salveaza sesiunea
        save_session_to_disk(session_id, session[session_id])

        # Citeste dimensiunile imaginii pentru frontend
        img = cv2.imread(original_filepath)
        height, width = img.shape[:2]

        return jsonify({
            'success': True,
            'message': 'Istoric sters si resetat la imaginea originala',
            'image': f"/uploads/{os.path.basename(original_filepath)}",
            'width': width,
            'height': height
        })

    except Exception as e:
        return jsonify({'error': f'Eroare la stergerea istoricului: {str(e)}'}), 500


# HTML Template Imbunatatit cu debugging
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Processor - Completely Fixed</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .status-bar {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }

        .main-content {
            display: grid;
            grid-template-columns: 300px 1fr 300px;
            gap: 20px;
            margin-bottom: 20px;
        }

        .sidebar {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }

        .image-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            min-height: 600px;
            display: flex;
            flex-direction: column;
        }

        .image-display {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #ddd;
            border-radius: 10px;
            background: #f9f9f9;
            margin-bottom: 20px;
            position: relative;
            overflow: auto;
        }

        .image-display img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }

        .upload-area {
            padding: 40px;
            color: #666;
            text-align: center;
        }

        .section-title {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #444;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            margin: 5px;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-small {
            padding: 8px 16px;
            font-size: 12px;
        }

        .btn-danger {
            background: linear-gradient(135deg, #ff416c 0%, #ff4757 100%);
        }

        .btn-success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }

        .controls-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }

        .filter-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
        }

        .input-group {
            margin-bottom: 15px;
        }

        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .history-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #f9f9f9;
        }

        .history-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background 0.2s;
        }

        .history-item:hover {
            background: #e3f2fd;
        }

        .history-item.active {
            background: #2196f3;
            color: white;
        }

        .tabs {
            display: flex;
            margin-bottom: 20px;
            background: #f0f0f0;
            border-radius: 8px;
            overflow: hidden;
        }

        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
            background: transparent;
        }

        .tab.active {
            background: #667eea;
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transform: translateX(400px);
            transition: transform 0.3s ease;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }

        .notification.error {
            background: linear-gradient(135deg, #ff416c 0%, #ff4757 100%);
        }

        .notification.info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .debug-info {
            font-size: 11px;
            color: #888;
            margin-top: 10px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Image Processor - COMPLETELY FIXED</h1>
            <p>Colorare AI • Filtre Cumulative • Istoric Persistent • Debug Imbunatatit</p>
        </div>

        <div class="status-bar">
            <div id="status-text">Se Incarca modelele AI...</div>
        </div>

        <div class="main-content">
            <!-- Sidebar stânga -->
            <div class="sidebar">
                <div class="section-title">Incarcare Imagine</div>
                <input type="file" id="image-input" accept="image/*" style="display: none;">
                <button class="btn" onclick="document.getElementById('image-input').click()">
                    Selecteaza Imagine
                </button>

                <div class="section-title" style="margin-top: 30px;">Procesare Principala</div>
                <div class="controls-grid">
                    <button class="btn" id="colorize-btn" onclick="colorizeImage()">Coloreaza AI</button>
                    <button class="btn" id="detect-btn" onclick="detectObjects()">Detecteaza Obiecte</button>
                </div>

                <div class="input-group">
                    <label for="scale-select">Marire Rezolutie:</label>
                    <select id="scale-select">
                        <option value="1.5">1.5x</option>
                        <option value="2" selected>2x</option>
                        <option value="3">3x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
                <button class="btn" id="upscale-btn" onclick="upscaleImage()">Mareste Rezolutia</button>

                <div class="section-title" style="margin-top: 30px;">Actiuni</div>
                <div class="controls-grid">
                    <div style="position: relative; display: inline-block;">
                        <button type="button" class="btn btn-success" id="save-btn" onclick="toggleMenu()">Salveaza</button>
                        <div id="format-menu" class="dropdown-content" style="display: none; position: absolute; top: 100%; left: 0; background: white; border: 1px solid #ccc; box-shadow: 0 2px 8px rgba(0,0,0,0.15); border-radius: 5px; z-index: 1000; min-width: 120px; margin-top: 5px;">
                            <a href="#" onclick="saveImage('png'); return false;" style="display: block; padding: 8px 12px; color: #333; text-decoration: none;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">PNG</a>
                            <a href="#" onclick="saveImage('jpeg'); return false;" style="display: block; padding: 8px 12px; color: #333; text-decoration: none;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">JPEG</a>
                            <a href="#" onclick="saveImage('jpg'); return false;" style="display: block; padding: 8px 12px; color: #333; text-decoration: none;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">JPG</a>
                            <a href="#" onclick="saveImage('gif'); return false;" style="display: block; padding: 8px 12px; color: #333; text-decoration: none;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">GIF</a>
                            <a href="#" onclick="saveImage('webp'); return false;" style="display: block; padding: 8px 12px; color: #333; text-decoration: none;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">WEBP</a>
                        </div>
                    </div>
                    <button class="btn btn-danger" onclick="resetImage()">Reset</button>
                </div>
            </div>
            <!-- Container central - Imagine -->
            <div class="image-container">
                <div class="image-display" id="image-display">
                    <div class="upload-area">
                        <h3>Gliseaza imaginea aici</h3>
                        <p>sau foloseste butonul "Selecteaza Imagine"</p>
                        <p style="font-size: 12px; margin-top: 10px;">
                            IMBUNaTatIRI: Filtre cumulative + Istoric persistent
                        </p>
                    </div>
                </div>
            </div>

            <!-- Sidebar dreapta -->
            <div class="sidebar">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('filters')">Filtre</button>
                    <button class="tab" onclick="switchTab('history')">Istoric</button>
                </div>

                <!-- Tab Filtre -->
                <div class="tab-content active" id="filters-tab">
                    <div class="section-title">Filtre Culoare</div>
                    <div class="filter-grid">
                        <button class="btn btn-small" onclick="applyFilter('red_channel')">Canal Rosu</button>
                        <button class="btn btn-small" onclick="applyFilter('green_channel')">Canal Verde</button>
                        <button class="btn btn-small" onclick="applyFilter('blue_channel')">Canal Albastru</button>
                        <button class="btn btn-small" onclick="applyFilter('grayscale')">Grayscale</button>
                        <button class="btn btn-small" onclick="applyFilter('sepia')">Sepia</button>
                    </div>

                    <div class="section-title" style="margin-top: 20px;">Efecte Speciale</div>
                    <div class="filter-grid">
                        <button class="btn btn-small" onclick="applyFilter('cartoon')">Cartoon</button>
                        <button class="btn btn-small" onclick="applyFilter('vintage')">Vintage</button>
                        <button class="btn btn-small" onclick="applyFilter('negative')">Negativ</button>
                        <button class="btn btn-small" onclick="applyFilter('edge_detection')">Detectare Margini</button>
                        <button class="btn btn-small" onclick="applyFilter('emboss')">Emboss</button>
                        <button class="btn btn-small" onclick="applyFilter('sharpen')">Ascutire</button>
                    </div>

                    <div class="section-title" style="margin-top: 20px;">Blur Control</div>
                    <div class="input-group">
                        <label for="blur-intensity">Intensitate:</label>
                        <input type="range" id="blur-intensity" min="1" max="50" value="15">
                        <span id="blur-value">15</span>
                    </div>
                    <button class="btn btn-small" onclick="applyBlur()">Aplica Blur</button>

                    <div class="section-title" style="margin-top: 20px;">Luminozitate</div>
                    <div class="input-group">
                        <label for="brightness-intensity">Intensitate:</label>
                        <input type="range" id="brightness-intensity" min="-100" max="100" value="0">
                        <span id="brightness-value">0</span>
                    </div>
                    <button class="btn btn-small" onclick="applyBrightness()">Aplica Luminozitate</button>

                    <!-- Font Awesome (include in <head> pentru a fi sigur ca e Incarcat) -->
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">

                    <div class="section-title" style="margin-top: 20px;">Rotire Imagine</div>
                    <div style="display: flex; gap: 10px; margin-top: 10px; align-items: center;">
                        <button class="btn btn-secondary btn-small" onclick="rotateImage(-90)" title="Rasuceste stânga">
                            <i class="fas fa-undo"></i>
                        </button>
                        <button class="btn btn-secondary btn-small" onclick="rotateImage(90)" title="Rasuceste dreapta">
                            <i class="fas fa-redo"></i>
                        </button>
                    </div>

                </div>

                <!-- Tab Istoric -->
                <div class="tab-content" id="history-tab">
                    <div class="section-title">Istoric Efecte</div>
                    <div class="history-list" id="history-list">
                        <div style="padding: 20px; text-align: center; color: #666;">
                            Nu exista istoric
                        </div>
                    </div>

                    <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
                        <button class="btn btn-danger btn-small" onclick="clearHistory()">
                            <i class="fas fa-trash-alt"></i> sterge Istoric
                        </button>
                        <button class="btn btn-secondary btn-small" onclick="undoHistory()">
                            <i class="fas fa-undo"></i> Undo
                        </button>
                        <button class="btn btn-secondary btn-small" onclick="redoHistory()">
                            <i class="fas fa-redo"></i> Redo
                        </button>
                    </div>
                </div>
                <!-- Font Awesome (daca nu Il ai deja inclus) -->
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
            </div>
        </div>
    </div>

    <div id="notification" class="notification"></div>

    <script>
        let currentSessionId = null;
        let isLoading = false;

        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
            setupEventListeners();
        });

        function setupEventListeners() {
            // Slider Blur
            const blurSlider = document.getElementById('blur-intensity');
            const blurValue = document.getElementById('blur-value');
            blurSlider.addEventListener('input', function() {
                blurValue.textContent = this.value;
            });

            // Slider Luminozitate
            const brightnessSlider = document.getElementById('brightness-intensity');
            const brightnessValue = document.getElementById('brightness-value');
            brightnessSlider.addEventListener('input', function() {
                brightnessValue.textContent = this.value;
            });
             document.getElementById('image-input').addEventListener('change', function(e) {
                if (e.target.files[0]) {
                    uploadImage(e.target.files[0]);
                }
            });
            // Rotire imagine
            const rotateLeftBtn = document.getElementById('rotate-left');
            const rotateRightBtn = document.getElementById('rotate-right');

            rotateLeftBtn.addEventListener('click', () => {
                rotateImage(-90);
            });

            rotateRightBtn.addEventListener('click', () => {
                rotateImage(90);
            });
            // Input imagine
        }



        function updateDebugInfo(message) {
            const debugDiv = document.getElementById('debug-info');
            if (debugDiv) {
                debugDiv.textContent = `Debug: ${message}`;
            } else {
                console.warn(`Debug info element not found. Message: ${message}`);
            }
        }
        function showNotification(message, type = 'info') {
            console.log(`Notification: ${message} (${type})`);
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type}`;
            notification.classList.add('show');

            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
        function toggleMenu() {
            const menu = document.getElementById("format-menu");
            if (!menu) {
                console.warn("Elementul 'format-menu' nu a fost gasit!");
                return;
            }
        
            const currentDisplay = window.getComputedStyle(menu).display;
            menu.style.display = (currentDisplay === "none") ? "block" : "none";
        
            // Debug: trimite log la server
            const logMessage = `toggleMenu apelata. Element ID: ${menu.id}`;
            console.log(logMessage);
        
            // Trimite log la /debug-save
            fetch('/debug-save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: logMessage, timestamp: Date.now() })
            }).catch(err => console.error('Eroare la trimiterea log-ului:', err));
        }

       async function undoHistory() {
            if (!currentSessionId) return;

            try {
                const response = await fetch('/api/history/undo', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });
                const data = await response.json();

                if (data.success && data.filepath) {
                    displayImage(data.image);
                    updateHistory();
                } else {
                    alert(data.error || "Nu se poate face undo");
                }
            } catch (err) {
                console.error("Eroare la undo:", err);
            }
        }

        async function redoHistory() {
            if (!currentSessionId) return;

            try {
                const response = await fetch('/api/history/redo', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });
                const data = await response.json();

                if (data.success && data.filepath) {
                   displayImage(data.image);
                    updateHistory();
                } else {
                    alert(data.error || "Nu se poate face redo");
                }
            } catch (err) {
                console.error("Eroare la redo:", err);
            }
        }


        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status-text').textContent = data.message;

                if (data.status === 'loading') {
                    setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                console.error('Eroare la verificarea statusului:', error);
            }
        }

        async function uploadImage(file) {
            if (isLoading) return;

            console.log('Incepe upload-ul imaginii:', file.name);
            setLoading(true);
            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                console.log('Raspuns upload:', data);

                if (data.success) {
                    currentSessionId = data.session_id;
                    console.log('Session ID setat:', currentSessionId);
                    updateDebugInfo(`Sesiune activa: ${currentSessionId.substring(0, 8)}...`);
                    displayImage(data.image);
                    enableControls();
                    updateHistory();
                    showNotification('Imagine Incarcata cu succes!', 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la upload:', error);
                showNotification('Eroare la Incarcarea imaginii', 'error');
            } finally {
                setLoading(false);
            }
        }

        function displayImage(imageSrc) {
            console.log('Afisare imagine:', imageSrc);
            const imageDisplay = document.getElementById('image-display');
            imageDisplay.innerHTML = `<img src="${imageSrc}?t=${Date.now()}" id="main-image">`;
        }

        function enableControls() {
            const buttons = ['colorize-btn', 'detect-btn', 'upscale-btn', 'save-btn'];
            buttons.forEach(id => {
                document.getElementById(id).disabled = false;
            });
        }

        function setLoading(loading) {
            isLoading = loading;
            const container = document.querySelector('.container');
            if (loading) {
                container.style.opacity = '0.6';
                container.style.pointerEvents = 'none';
            } else {
                container.style.opacity = '1';
                container.style.pointerEvents = 'auto';
            }
        }

        async function colorizeImage() {
            if (!currentSessionId || isLoading) {
                showNotification('Nu exista sesiune activa!', 'error');
                return;
            }

            console.log('Colorize pentru sesiunea:', currentSessionId);
            setLoading(true);
            updateDebugInfo('Se aplica colorizarea...');

            try {
                const response = await fetch('/api/colorize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });

                const data = await response.json();
                console.log('Raspuns colorize:', data);

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    updateDebugInfo('Colorizare aplicata cu succes');
                    showNotification('Colorizare aplicata cu succes!', 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                    updateDebugInfo('Eroare la colorizare: ' + data.error);
                }
            } catch (error) {
                console.error('Eroare la colorizare:', error);
                showNotification('Eroare la colorizarea imaginii', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function applyFilter(filterType, angle = 0) {
            if (!currentSessionId || isLoading) {
                showNotification('Nu exista sesiune activa!', 'error');
                return;
            }

            console.log('Aplicare filtru:', filterType, 'pentru sesiunea:', currentSessionId);
            setLoading(true);
            updateDebugInfo(`Se aplica filtrul ${filterType}...`);

            try {
                 const bodyData = { session_id: currentSessionId, filter_type: filterType };
                // Daca filtru = rotate, trimitem angle
                if (filterType === 'rotate') {
                    bodyData.params = { angle: angle };
        }

                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        filter_type: filterType
                    })
                });

                const data = await response.json();
                console.log('Raspuns filtru:', data);

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    updateDebugInfo(`Filtru ${filterType} aplicat cu succes`);
                    showNotification(`Filtru ${filterType} aplicat!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                    updateDebugInfo('Eroare la filtru: ' + data.error);
                }
            } catch (error) {
                console.error('Eroare la aplicarea filtrului:', error);
                showNotification('Eroare la aplicarea filtrului', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function applyBlur() {
            const intensity = document.getElementById('blur-intensity').value;
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        filter_type: 'blur',
                        params: {strength: parseInt(intensity)}
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    showNotification(`Blur aplicat cu intensitatea ${intensity}!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la aplicarea blur-ului:', error);
                showNotification('Eroare la aplicarea blur-ului', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function rotateImage(degrees) {
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        filter_type: 'rotate',
                        params: {angle: degrees}
                    })
                });

                const data = await response.json();

                if (data.success) {
                   displayImage(data.image)

                    updateHistory();
                    showNotification(`Imagine rotita cu ${degrees}°!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la rotirea imaginii:', error);
                showNotification('Eroare la rotirea imaginii', 'error');
            } finally {
                setLoading(false);
            }
        }





        async function applyBrightness() {
            const intensity = document.getElementById('brightness-intensity').value;
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        filter_type: 'brightness',
                        params: {intensity: parseInt(intensity)}
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);        // actualizeaza imaginea
                    updateHistory();                 // actualizeaza istoricul
                    showNotification(`Luminozitate aplicata cu intensitatea ${intensity}!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la aplicarea luminozitatii:', error);
                showNotification('Eroare la aplicarea luminozitatii', 'error');
            } finally {
                setLoading(false);
            }
        }


        async function detectObjects() {
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    showNotification(`${data.objects.length} obiecte detectate!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la detectarea obiectelor:', error);
                showNotification('Eroare la detectarea obiectelor', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function upscaleImage() {
            const scaleFactor = document.getElementById('scale-select').value;
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/upscale', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        scale_factor: parseFloat(scaleFactor)
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    showNotification(`Imaginea a fost marita ${scaleFactor}x!`, 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la marirea imaginii:', error);
                showNotification('Eroare la marirea imaginii', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function resetImage() {
            if (!currentSessionId || isLoading) return;

            if (!confirm('Sigur doriti sa resetati la imaginea originala?')) return;

            setLoading(true);

            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);
                    updateHistory();
                    showNotification('Imaginea a fost resetata!', 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la resetare:', error);
                showNotification('Eroare la resetarea imaginii', 'error');
            } finally {
                setLoading(false);
            }
        }

        function saveImage(format) {
            if (!currentSessionId) return;
        
            const img = document.getElementById('main-image');
            
            if (!img) {
                showNotification('Nu există imagine de salvat!', 'error');
                return;
            }
        
            // Creeaza un canvas pentru conversie
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Seteaza dimensiunile canvas-ului
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            
            // Deseneaza imaginea pe canvas
            ctx.drawImage(img, 0, 0);
            
            // Determina MIME type și calitate
            let mimeType, quality = 0.95;
            
            switch(format) {
                case 'png':
                    mimeType = 'image/png';
                    break;
                case 'jpeg':
                case 'jpg':
                case 'jfif':
                    mimeType = 'image/jpeg';
                    quality = 0.92;
                    break;
                case 'svg':
                    showNotification('SVG nu este suportat pentru imagini raster. Alegeți PNG sau JPEG.', 'error');
                    return;
                default:
                    mimeType = 'image/png';
            }
            
            // Converteste canvas la blob și descarca
            try {
                canvas.toBlob(function(blob) {
                    if (!blob) {
                        showNotification('Eroare la salvarea imaginii!', 'error');
                        return;
                    }
                    
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.download = `processed_image.${format}`;
                    link.href = url;
                    link.click();
                    
                    // Elibereaza memoria
                    URL.revokeObjectURL(url);
                    
                    showNotification(`Imaginea a fost salvată ca ${format.toUpperCase()}!`, 'success');
                }, mimeType, quality);
            } catch (error) {
                console.error('Eroare la salvare:', error);
                showNotification('Eroare la salvarea imaginii!', 'error');
            }
        }

        async function updateHistory() {
            if (!currentSessionId) return;

            console.log('Actualizare istoric pentru sesiunea:', currentSessionId);

            try {
                const response = await fetch('/api/history', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: currentSessionId})
                });

                const data = await response.json();
                console.log('Raspuns istoric:', data);

                if (data.success) {
                    displayHistory(data.history);
                    updateDebugInfo(`Istoric actualizat: ${data.history.length} intrari`);
                }
            } catch (error) {
                console.error('Eroare la actualizarea istoricului:', error);
            }
        }

        function displayHistory(history) {
            const historyList = document.getElementById('history-list');

            if (history.length === 0) {
                historyList.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">Nu exista istoric</div>';
                return;
            }

            historyList.innerHTML = history.map((item, index) => {
                const date = new Date(item.timestamp * 1000);
                const timeStr = date.toLocaleTimeString('ro-RO', {hour: '2-digit', minute: '2-digit'});

                return `<div class="history-item ${index === history.length - 1 ? 'active' : ''}" onclick="goToHistory(${index})">
                    ${index + 1}. ${item.name} (${timeStr})
                </div>`;
            }).join('');

            console.log('Istoric afisat cu', history.length, 'intrari');
        }

        async function goToHistory(index) {
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/history/goto', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        index: index
                    })
                });

                const data = await response.json();

                if (data.success) {
                    displayImage(data.image);
                    document.querySelectorAll('.history-item').forEach((item, i) => {
                        item.classList.toggle('active', i === index);
                    });
                    showNotification('Navigare In istoric completa!', 'success');
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la navigarea In istoric:', error);
                showNotification('Eroare la navigarea In istoric', 'error');
            } finally {
                setLoading(false);
            }
        }

        async function clearHistory() {
            if (!confirm('Sigur doriti sa stergeti istoricul?')) return;
            if (!currentSessionId || isLoading) return;

            setLoading(true);

            try {
                const response = await fetch('/api/history/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessionId })
                });

                const data = await response.json();

                if (data.success) {
                    // Apel reset dupa clear
                    const resetResponse = await fetch('/api/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: currentSessionId })
                    });

                    const resetData = await resetResponse.json();

                    if (resetData.success) {
                        displayImage(resetData.image);    

                        updateHistory();  // actualizeaza lista de istoric
                        showNotification('Istoric sters si imagine resetata!', 'success');
                    } else {
                        showNotification('Clear ok, dar reset a esuat: ' + resetData.error, 'error');
                    }
                } else {
                    showNotification('Eroare: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Eroare la stergerea istoricului:', error);
                showNotification('Eroare la stergerea istoricului', 'error');
            } finally {
                setLoading(false);
            }
        }
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }
    </script>
</body>
</html>"""


def create_templates_dir():
    """Creeaza directorul templates si fisierul index.html"""
    if not os.path.exists(TEMPLATES_FOLDER):
        os.makedirs(TEMPLATES_FOLDER)

    with open(os.path.join(TEMPLATES_FOLDER, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)


def create_requirements_file():
    """Creeaza fisierul requirements.txt"""
    requirements = """flask==2.3.3
opencv-python==4.8.1.78
numpy==1.24.3
requests==2.31.0
Pillow==10.0.1
scikit-image==0.21.0
werkzeug==2.3.7
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)


if __name__ == '__main__':
    print("Initializare AI Image Processor - PRODUCTION READY VERSION")
    create_templates_dir()
    create_requirements_file()
    # Încarcă modelele **sincron**, astfel încât logurile să fie vizibile
    #load_models()

    # Pornește serverul Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # debug=True doar local
        threaded=True,
        use_reloader=False
    )