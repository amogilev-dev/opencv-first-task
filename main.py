import cv2
import os

# пути к файлам
folder_dir = "./faces"
results_dir = "./results"

# загрузка модели, эта модель различает лица и в профиль, и в анфас
model_path = "./models/face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(
    model_path,             # путь к файлу с моделью
    "",                     # конфиг файл, не нужен для этой модели
    (0, 0),                 # размер картинки, устанавливается динамически
    score_threshold=0.8,    # уверенность в том есть ли лицо
    nms_threshold=0.3,      # насколько рамки могут перекрываться, иначе одно лицо считается дважды
    top_k=5000,             # сколько кандидатов в лица модель рассматривает до фильтрации
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,  # чем запускать нейросеть, сейчас встроенный движок OpenCV
    target_id=cv2.dnn.DNN_TARGET_CPU,       # на каком устройстве считать, сейчас установлен процессор
)

with_faces = []
without_faces = []

def draw_rectangle(img, rect):
    (x, y, w, h) = map(int, rect[:4])
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
    os.makedirs(results_dir, exist_ok=True)

    for filename in os.listdir(folder_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder_dir, filename)

        image = cv2.imread(path)
        if image is None:
            print(f"error reading file: {filename}")
            continue

        # модели нужно знать размер картинки
        height, width = image.shape[:2]
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)

        if faces is None:
            without_faces.append(filename)
            continue

        with_faces.append(filename)

        for face in faces:
            draw_rectangle(image, face)

        result_path = os.path.join(results_dir, filename)
        if not cv2.imwrite(result_path, image):
            print(f"error saving file: {filename}")

    for name in with_faces:
        print("+   ", name)

    for name in without_faces:
        print("-   ", name)

if __name__ == '__main__':
   main()

