from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


# Pydantic schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person")
    city: str = Field(description="City where the person lives")
    description:str=Field(description="Brief summary about the person")


# Parser connected to schema
parser = JsonOutputParser(pydantic_object=Person)


# Prompt
prompt = PromptTemplate(
    template="""
    Random famous female porn actress.
{format_instructions}
""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# Chain
chain = prompt | model | parser


# Run
result = chain.invoke({})

print(result)
print(type(result))

# to visualize chain
chain.get_graph().print_ascii()