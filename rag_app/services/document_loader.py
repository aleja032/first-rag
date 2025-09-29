from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# # Example documents
# docs = [
#     Document(page_content="This is a test document about LangChain."),
#     Document(page_content="Another document about AI and embeddings."),
# ]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in docs:
    chunks.extend(text_splitter.split_text(doc.page_content))

# Add chunks to vector store
vector_store.add_documents([Document(page_content=chunk) for chunk in chunks])
