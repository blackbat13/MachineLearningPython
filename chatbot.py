from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize model (first download ~2GB)
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

# Create chat pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Chat loop
print("Chatbot: Hi! Ask me anything (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    
    # Generate response
    messages = [{"role": "user", "content": user_input}]
    response = chatbot(messages, max_new_tokens=150, do_sample=True)[0]['generated_text'][-1]
    
    print(f"\nChatbot: {response['content']}")