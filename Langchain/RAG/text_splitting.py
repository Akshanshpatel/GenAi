# Length based text splitting --> Fast simple but cut words in b/w so not used

# Text Structure based
# RecursiveCharacterTextSplitting  ((\n\n) para-> (\n) line -> '_' word -> '' character)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text= """In LangChain, Models are the core reasoning engines.
They act as standardized abstraction layers that allow your application to seamlessly interact with various AI providers
(like OpenAI, Anthropic, and Google) without rewriting your code. LangChain primarily features three types of models.1.
Chat ModelsWhat they do: Optimized for multi-turn conversations and reasoning, these models power AI assistants.Input/Output:
They accept structured messages (e.g., a list of SystemMessage, HumanMessage, and AIMessage) and return response messages.

Examples: GPT-4o, Claude, and Gemini.2. Large Language Models (LLMs)What they do: These are foundational text-completion
models.Input/Output: They accept a single raw text string as input and generate a plain text string as output.
Examples: Legacy text-completion models. (While supported in LangChain, modern apps primarily rely on Chat Models).
3. Text Embedding ModelsWhat they do: Used primarily in retrieval and search, these models do not generate text.
Instead, they convert text into numerical arrays (vectors) that represent the semantic meaning of the words.
Input/Output: Text goes in; a list of floating-point numbers comes out.Examples: OpenAI text-embedding-3 or
Google text-embedding-004.Why Use LangChain Models?Interoperability: You can write your core application logic and instantly swap the
underlying AI provider by just changing a parameter.Feature Integration: LangChain provides out-of-the-box support for
advanced model features like tool calling, structured JSON outputs, and multimodal processing."""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)

# Document Structure based
# we use Recursive one here too but splitting technique differs

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
text="""Some python code here"""

splitter=RecursiveCharacterTextSplitter(
    Language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)
# will make chunks based on python language --> class functions separately


# Semantic Based Splitting is self-explanatory