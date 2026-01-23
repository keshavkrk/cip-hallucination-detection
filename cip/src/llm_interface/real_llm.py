from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def llm_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer factually and concisely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()
