from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


#Text Loader
from langchain_community.document_loaders import TextLoader
loader=TextLoader('path-of-file')
doc1=loader.load()


model=ChatGroq(model="llama-3.3-70b-versatile")
parser=StrOutputParser()

#pyPdfLoader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('Akshansh_Btech_GLAU.pdf')
doc=loader.load()

pdf_text=doc[0].page_content

prompt=PromptTemplate(
    template="""Based on the resume {text} Tell me about {topic} in brief""",
    input_variables=['text','topic']
)

chain=prompt | model | parser

print(chain.invoke({'text':pdf_text,'topic':'Akshansh Patel'}))

## this pypdfLoader works well on textual pdf but not others(images etc)
## for those-->  https://docs.langchain.com/oss/python/integrations/document_loaders

# For directory loading--> DirectoryLoader


# WebBaseLoader  --> Good for static websites (Uses request and BeautifulSoup internally)