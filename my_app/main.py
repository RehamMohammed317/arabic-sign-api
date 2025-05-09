from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
gesture_model = GestureRecognizer.create_from_model_path("/content/gesture_recognizer.task")
words_model = tf.lite.Interpreter(model_path="/content/arsl_word_model.tflite")
words_model.allocate_tensors()
numbers_model = tf.lite.Interpreter(model_path="/content/sign_language_model.tflite")
numbers_model.allocate_tensors()

# Arabic labels
letter_labels = {
    "Alef": "ا", "Beh": "ب", "Teh": "ت", "Theh": "ث", "Jeem": "ج", "Hah": "ح",
    "Khah": "خ", "Dal": "د", "Thal": "ذ", "Reh": "ر", "Zain": "ز", "Seen": "س",
    "Sheen": "ش", "Sad": "ص", "Dad": "ض", "Tah": "ط", "Zah": "ظ", "Ain": "ع",
    "Ghain": "غ", "Feh": "ف", "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Meem": "م",
    "Noon": "ن", "Heh": "ه", "Waw": "و", "Yeh": "ي", "Teh_Marbuta": "ة", "Laa": "لا", "Al": "ال"
}

word_labels = {
     "club": "نادي", "father": "اب", "help": "يساعد", "learn": "يتعلم", "love": "يحب",
    "mother": "ام", "school": "مدرسة", "sorry": "اسف", "thanks": "شكرا", "where": "اين"
}

number_labels = {i: str(i) for i in range(10)}

def preprocess_image(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

@app.post("/predict/letters/")
async def predict_letters(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = gesture_model.recognize(mp_image)
    top = result.gestures[0][0].category_name if result.gestures else None
    return {"prediction": letter_labels.get(top, "غير معروف")}

@app.post("/predict/words/")
async def predict_words(file: UploadFile = File(...)):
    contents = await file.read()
    image = preprocess_image(contents)
    input_details = words_model.get_input_details()
    output_details = words_model.get_output_details()
    words_model.set_tensor(input_details[0]['index'], image)
    words_model.invoke()
    output = words_model.get_tensor(output_details[0]['index'])
    prediction = word_labels[np.argmax(output)]
    return {"prediction": prediction}

@app.post("/predict/numbers/")
async def predict_numbers(file: UploadFile = File(...)):
    contents = await file.read()
    image = preprocess_image(contents)
    input_details = numbers_model.get_input_details()
    output_details = numbers_model.get_output_details()
    numbers_model.set_tensor(input_details[0]['index'], image)
    numbers_model.invoke()
    output = numbers_model.get_tensor(output_details[0]['index'])
    prediction = number_labels[np.argmax(output)]
    return {"prediction": prediction}
