import os
from dotenv import load_dotenv
import requests

load_dotenv()

def getInterpretation(prediction, prob_malignant):
    API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }

    diagnosis = "ganas (malignant)" if prediction == 1 else "jinak (benign)"
    prob_percent = round(prob_malignant * 100, 2)
    prompt = (
        f"Saya memiliki hasil prediksi kanker payudara menggunakan model machine learning logistic regression yang dilatih pada data Breast Cancer Wisconsin. "
        f"Model ini menganalisis hasil pencitraan Fine Needle Aspiration sel-sel tumor. "
        f"Hasil prediksi menunjukkan tumor {diagnosis} dengan probabilitas ganas {prob_percent}%. "
        f"Tolong berikan interpretasi analitis atas hasil ini dalam bahasa Indonesia yang mudah dipahami pasien. "
        f"Jelaskan apa arti hasil prediksi dan probabilitas tersebut. Jawab straight to the point"
    )

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    response = query({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct"
    })

    return response["choices"][0]["message"]['content']