"""
RAG test

reference:
https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2

"""
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Weaviate

from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import weaviate
import warnings
warnings.filterwarnings("ignore")


# load and chunk a text file with context info for RAG
loader = TextLoader('./rag_context.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# set up a vector store
client = weaviate.Client( embedded_options = EmbeddedOptions())
vectorstore = Weaviate.from_documents(client = client, documents = chunks,
                                      embedding = OpenAIEmbeddings(), by_text = False)
retriever = vectorstore.as_retriever()


# set up the prompt template and RAG chain
template="""{user_query}\nContext:{context}"""
prompt_template = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever,  "user_query": RunnablePassthrough()} 
    | prompt_template 
    | llm
    | StrOutputParser() 
)


# loop to get user queries
print(f"\n\nReady for User Queries...")
while True:
    user_query = input()
    response = rag_chain.invoke(user_query) 
    print(response)
    print("\n\n")


