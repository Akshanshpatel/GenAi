from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnables import RunnableSequence   --> not ideal now
from langchain_core.runnables import RunnableSequence

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

prompt1=PromptTemplate(
    template="Crack a joke about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain1=RunnableSequence(prompt1,model,parser)

print(chain1.invoke({'topic':'India'}))


prompt2=PromptTemplate(
    template="Explain the joke in brief {joke}",
    input_variables=['joke']
)

chain2=RunnableSequence(prompt1,model,parser,{"joke":lambda x:x},prompt2,model,parser)

print(chain2.invoke({'topic':'India'}))