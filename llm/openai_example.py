# pip install openai

from openai import OpenAI

client = OpenAI(api_key="YOUR API KEY")


response = client.responses.create(
    model="gpt-4o",
    input="Explain quantum computing in simple terms."
)

print(response.output_text)
