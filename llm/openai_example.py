# pip install openai

import openai

def ask_llm(question):
    openai.api_key = "sk-your-api-key-here"  # Replace with your actual API key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message['content']

if __name__ == "__main__":
    prompt = "Explain quantum computing in simple terms"
    answer = ask_llm(prompt)
    print(f"Q: {prompt}\nA: {answer}")