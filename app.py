from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import pickle
import imgaug.augmenters as iaa
import os
from werkzeug.utils import secure_filename
import mediapipe as mp
from collections import deque
from scipy.spatial.distance import cosine

# Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'face_models'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Initialize ML components
embedder = FaceNet()
detector = MTCNN()
mp_pose = mp.solutions.pose


# ==========================
# Face Model Builder
# ==========================
class FaceModelBuilder:
    def __init__(self, image_path, uid, output_dir):
        self.image_path = image_path
        self.uid = uid
        self.output_dir = os.path.join(output_dir, uid)
        self.embedder = embedder
        self.detector = detector
        self.embeddings_dict = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def extract_face_from_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)

        if not faces:
            raise ValueError("No face detected")

        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        face_img = rgb_image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        return face_img

    def augment_face_data(self, face_img, num_augmentations=30):
        augmenters = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),
            iaa.Sometimes(0.5, iaa.Affine(scale=(0.95, 1.05))),
            iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.2, iaa.Fliplr()),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(5, 15))),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.5, 1.0))),
        ])
        augmented_faces = [face_img]
        for i in range(num_augmentations):
            augmented_faces.append(augmenters(image=face_img))
        return augmented_faces

    def extract_embeddings(self, face_images):
        embeddings = []
        for i, face_img in enumerate(face_images):
            try:
                embedding = self.embedder.embeddings(np.array([face_img]))[0]
                embeddings.append(embedding)
                self.embeddings_dict[f"face_variant_{i}"] = embedding
            except Exception as e:
                print(f"Embedding failed for variant {i}: {e}")
        return embeddings

    def save_model(self):
        model_path = os.path.join(self.output_dir, "face_embeddings.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.embeddings_dict, f)
        return model_path

    def build_model(self, num_augmentations=30):
        try:
            face_img = self.extract_face_from_image()
            augmented_faces = self.augment_face_data(face_img, num_augmentations)
            embeddings = self.extract_embeddings(augmented_faces)
            model_path = self.save_model()
            return {"success": True, "model_path": model_path, "embeddings_count": len(embeddings)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ==========================
# Video Analyzer
# ==========================
class VideoAnalyzer:
    def __init__(self, video_path, face_model_path):
        self.video_path = video_path
        with open(face_model_path, 'rb') as f:
            self.embeddings_dict = pickle.load(f)
        self.embedder = embedder
        self.detector = detector

    def extract_face_embedding(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_image)
            if not faces: return None
            largest_face = max(faces, key=lambda x: x['box'][2]*x['box'][3])
            x, y, w, h = largest_face['box']
            face_img = rgb_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            return self.embedder.embeddings(np.array([face_img]))[0]
        except:
            return None

    def authorize_face(self, frame, threshold=0.4):
        embedding = self.extract_face_embedding(frame)
        if embedding is None: return False, 0.0
        min_distance = min([cosine(embedding, e) for e in self.embeddings_dict.values()])
        similarity = 1 - min_distance
        return min_distance < threshold, similarity

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def analyze_video(self):
        cap = cv2.VideoCapture(self.video_path)
        situps, stage = 0, None
        angle_history, situp_details = deque(maxlen=5), []
        success_count, fail_count = 0, 0
        authorized, blocked = False, False
        frame_count, interval = 0, 30

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                if not authorized and not blocked and frame_count % interval == 0:
                    ok, sim = self.authorize_face(frame)
                    if ok:
                        success_count += 1
                        fail_count = 0
                        if success_count >= 3: authorized = True
                    else:
                        fail_count += 1
                        if fail_count >= 3: blocked = True; break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks and (authorized or success_count > 0):
                    lm = results.pose_landmarks.landmark
                    h, w, _ = frame.shape
                    r_sh, r_hp, r_kn = [lm[12].x*w, lm[12].y*h], [lm[24].x*w, lm[24].y*h], [lm[26].x*w, lm[26].y*h]
                    l_sh, l_hp, l_kn = [lm[11].x*w, lm[11].y*h], [lm[23].x*w, lm[23].y*h], [lm[25].x*w, lm[25].y*h]

                    torso_angle = (self.calculate_angle(r_sh, r_hp, r_kn) + self.calculate_angle(l_sh, l_hp, l_kn))/2
                    angle_history.append(torso_angle)
                    smooth_angle = np.mean(angle_history)

                    if smooth_angle > 130 and stage != "down": stage = "down"
                    elif smooth_angle < 115 and stage == "down":
                        situps += 1
                        situp_details.append({"rep": situps, "time": round(current_time,2), "angle": round(smooth_angle,1)})
                        stage = "up"

        cap.release()
        return {
            "success": True,
            "situps": situps,
            "authorized": authorized,
            "blocked": blocked,
            "success_attempts": success_count,
            "failed_attempts": fail_count,
            "situp_details": situp_details
        }


# ==========================
# Routes
# ==========================
@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'photo' not in request.files: return jsonify({"error": "No photo"}), 400
    uid = request.form.get("uid")
    if not uid: return jsonify({"error": "No UID"}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{uid}_face.jpg"))
    request.files['photo'].save(path)

    builder = FaceModelBuilder(path, uid, app.config['MODELS_FOLDER'])
    result = builder.build_model()
    return (jsonify({"message":"Model created","count":result["embeddings_count"]}),200 
            if result["success"] else (jsonify({"error":result["error"]}),500))


@app.route('/upload_profile', methods=['POST'])
def upload_profile():
    if 'photo' not in request.files: return jsonify({"error":"No photo"}),400
    uid = request.form.get("uid")
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{uid}_profile.jpg"))
    request.files['photo'].save(path)
    return jsonify({"photoUrl": f"{request.host_url}uploads/{os.path.basename(path)}"}),200


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files: return jsonify({"error":"No video"}),400
    uid = request.form.get("uid")
    if not uid: return jsonify({"error":"No UID"}),400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{uid}_video.mp4"))
    request.files['video'].save(video_path)

    face_model_path = os.path.join(app.config['MODELS_FOLDER'], uid, "face_embeddings.pkl")
    if not os.path.exists(face_model_path): return jsonify({"error":"Face model not found"}),404

    analyzer = VideoAnalyzer(video_path, face_model_path)
    result = analyzer.analyze_video()
    os.remove(video_path)
    return jsonify(result),200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
