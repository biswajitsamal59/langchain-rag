from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import bs4
from langchain_community.document_loaders import WebBaseLoader

from local_secrets import GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash", temperature=1)

import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)\
 
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

