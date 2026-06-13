import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def load_patients():
    with open("data/patients.json", "r") as f:
        return json.load(f)

def analyze_patient(patient: dict) -> dict:
    prompt = f"""
    You are a clinical decision support AI assistant.
    
    Analyze this patient data and respond in this EXACT format:
    
    SUMMARY: (2-3 sentence clinical summary)
    
    RISK FLAGS:
    - (list each critical concern)
    
    ACTIONS:
    1. (first priority action)
    2. (second priority action)
    3. (third priority action)
    
    RISK LEVEL: (choose one: LOW / MEDIUM / HIGH / CRITICAL)
    
    Patient Data:
    {json.dumps(patient, indent=2)}
    """

    url = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert clinical AI assistant. Be concise, accurate and actionable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }
    
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    result = response.json()
    
    return {
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "age": patient["age"],
        "analysis": result["choices"][0]["message"]["content"]
    }

def main():
    patients = load_patients()
    
    print("=" * 65)
    print("   HEALTHCARE AI AGENT — Patient Risk Analysis System")
    print("=" * 65)
    
    for patient in patients:
        print(f"\n📋 Processing: {patient['name']} (ID: {patient['patient_id']})")
        print("-" * 65)
        result = analyze_patient(patient)
        print(result["analysis"])
        print("=" * 65)

if __name__ == "__main__":
    main()