from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

# prompt 1 --> detailed report on a particular topic
template1=PromptTemplate(
    template='Write a detailed summary about the topic {topic}',
    input_variables=['topic']
)

# prompt 2--> 5 line summary of it
template2=PromptTemplate(
    template='Give a 3 line summary about {summarized_topic}',
    input_variables=['summarized_topic']
)

##This is normal result.content way without strOutputParser
"""
prompt1=template1.invoke({"topic":"Black holes"})

result1=model.invoke(prompt1)

prompt2=template2.invoke({"summarized_topic":result1.content})

final_output=model.invoke(prompt2)

print(final_output.content)
"""

parser1=StrOutputParser()

#main benefit of using StrOutputParser--> we can do it in one chain as parser gives result.content

## Also an example of Sequential chain
chain= template1 | model | parser1 | template2 | model | parser1

# print(chain.invoke({'topic':'CR7'}))

parser2=JsonOutputParser()

template3=PromptTemplate(
    template='Give name,city,age of a random famous person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

## Example of simple chain
chain2 = template3 | model | parser2

print(chain2.invoke({}))
print(type(chain2))