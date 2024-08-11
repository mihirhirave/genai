"""
This contains code required to run the RAG model using the gemini flash. To be run in a jupyter notebook.
This is just for documentation purposes.
"""
#1
%pip install llama-index-llms-gemini
%pip install llama-index llama-index-readers-web
%pip install llama-index-embeddings-huggingface
%pip install -q llama-index google-generativeai

%env GOOGLE_API_KEY="paste API KEY here"
import os

GOOGLE_API_KEY = "paste API KEY here"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = "paste API KEY here"
from llama_index.llms.gemini import Gemini

resp = Gemini().complete("Write a poem about a magic backpack")
print(resp)

#3
from llama_index.llms.gemini import Gemini
with open('mytext.txt', 'w') as writefile:
    writefile.write("Ravan won the ramayana and killed rama")
with open('mytext.txt', 'r') as testwritefile:
    print(testwritefile.read())
llm = Gemini()
Settings.llm = llm

from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import Settings


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model_bge = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#4
from llama_index.core.node_parser import SentenceSplitter

#Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
import os

link = "https://machine-learning.utdallas.edu/who-we-are/"

documents = SimpleWebPageReader(html_to_text=True).load_data(
    [link]
)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#documents = SimpleDirectoryReader(input_files=["mytext.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
from llama_index.core import Settings

#Settings.llm = AzureOpenAI(engine="my-custom-llm", model="gpt-4", temperature=0.0)
query_engine = index.as_query_engine()
response = query_engine.query("where did Dr. Yu Xiang worked before joining UTD")
print(response)


