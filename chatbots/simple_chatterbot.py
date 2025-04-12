# pip install chatterbot chatterbot_corpus

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a chatbot instance
chatbot = ChatBot(
    'MyBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3'
)

# Train with custom data
trainer = ListTrainer(chatbot)
custom_train_data = [
    "Hello",
    "Hi there! How can I help you?",
    "What's your name?",
    "I'm MyBot, your friendly chatbot!",
    "Goodbye",
    "See you later!"
]
trainer.train(custom_train_data)

# Train with English corpus data
corpus_trainer = ChatterBotCorpusTrainer(chatbot)
corpus_trainer.train(
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations"
)

# Chat interaction loop
print("MyBot: Hello! Type 'exit' to end the conversation.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.get_response(user_input)
        print(f"MyBot: {response}")
        
    except (KeyboardInterrupt, EOFError):
        break