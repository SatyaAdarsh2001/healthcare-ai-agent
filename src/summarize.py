import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

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
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {
                "role": "system",
                "content": "You are an expert clinical AI assistant. Be concise, accurate and actionable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=600
    )
    
    return {
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "age": patient["age"],
        "analysis": response.choices[0].message.content
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