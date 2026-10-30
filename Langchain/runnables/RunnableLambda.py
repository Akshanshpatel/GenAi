from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

def length(joke):
    return len(joke.split())

prompt=PromptTemplate(
    template="Crack a new random joke about the {topic}",
    input_variables=['topic']
)

chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'len_joke':RunnableLambda(length)
})

final_chain=RunnableSequence(chain,parallel_chain)

result=final_chain.invoke({'topic':'AI'})

final_result="""{} \nWord-count={}""".format(result['joke'],result['len_joke'])

print(final_result)

# similarly RunnableBranch is used for conditional branches (if-else of Langchain)