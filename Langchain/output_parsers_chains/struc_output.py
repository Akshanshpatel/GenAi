from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv("../.env")


#this one is simple TypedDict
class Review(TypedDict):
    summary:str
    sentiment:str

## this one is Annotated
# we are making this one as pros and cons list one
class crows(TypedDict):
    pros_cons:Annotated[str,"Give pros and cons about the product from the review namingly pros: and cons: if no pros or cons found just type None"]
    overall:Annotated[str,"Tell whether it has many pros or cons"]

# or we can use a json_schema or pydantic 
json_schema = {
    "title": "SummarySentimentResponse",
    "description": "Structured response containing a concise summary and sentiment analysis.",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A concise summary of the provided input text."
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
            "description": "Overall sentiment of the input text."
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score for the sentiment prediction."
        },
        "key_points": {
            "type": "array",
            "description": "Important points extracted from the text.",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["summary", "sentiment"]
}


model=ChatGroq(model="llama-3.3-70b-versatile")


structured_model=model.with_structured_output(json_schema)
# this is not magic that without any prompt model is giving summary and sentiment 
# a system prompt is generated when called with_structured_output

# user_input=input("Enter message : ")
ans=structured_model.invoke( """
OpenAI released a new AI model today that significantly improves reasoning
and coding capabilities. Early users reported faster responses and better
accuracy on complex tasks, although some developers mentioned higher API costs.
Overall, the launch received positive reactions from the AI community.
""")

print("\n\n")

print(ans)