from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv("../.env")

messages=[
    SystemMessage(content='You are an Ai Assistant ')
]

model=ChatGroq(model="llama-3.3-70b-versatile")

while True:
    user_input=input("You: ")
    messages.append(HumanMessage(content=user_input))
    if user_input=="quit": break

    result=model.invoke(user_input)
    messages.append(AIMessage(content=result.content))
    print("BOT: ",result.content)

print(messages)