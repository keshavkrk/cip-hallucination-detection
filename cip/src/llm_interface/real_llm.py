from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

BAD_PATTERNS = ["i don't know", "i do not know", "as an ai", "i cannot"]

def is_weak_answer(text: str) -> bool:
    t = text.lower()
    if len(t) < 20:
        return True
    if any(p in t for p in BAD_PATTERNS):
        return True
    return False

def llm_answer(prompt: str) -> str:
    # First attempt
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer factually and concisely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )

    answer = response.choices[0].message.content.strip()

    # Retry if weak
    if is_weak_answer(answer):
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Answer factually and directly. Do not say you don't know unless absolutely necessary."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()

    return answer
