import pymupdf
import pymupdf4llm
from pathlib import Path
import os
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
import hashlib
import chromadb
from chromadb import Collection

load_dotenv()
ACCEPTED_FILE_TYPES = [".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".svg", ".txt", ".doc", ".docx"]
SYSTEM_PROMPT = "You are a helpful, kind, and thoughtful assistant helping the user answer questions they have about a document.\
    You should do your best to answer the questions posed to you by the user, but if you do not know the answer, likely because \
    the answer is not in the information, you should let the user know that you do not have an answer but try to help the user by\
    telling them what information would be needed to answer the question. You should not give answers that are not relevant to the\
    context received or use outside information to guess at an answer. Stick strictly to the context provided."
CHUNK_THRESHOLD = 1.75
TOP_N = 5

# Convert document to markdown format in chunks
def doc_to_chunks(path: Path) -> list:
    if not path.exists(): 
        raise FileNotFoundError("Path does not exist")
    if not path.is_file(): 
        raise FileNotFoundError("Path is a directory, not a file")
    if path.suffix not in ACCEPTED_FILE_TYPES: 
        raise ValueError("Not a supported file type")
    doc = pymupdf.open(path)
    md_text = pymupdf4llm.to_markdown(doc)
    index = 0
    chunks = []
    while (index < len(md_text)):
        chunks.append(md_text[index: index + 1000])
        index += 800
    return chunks


def find_relevant_chunks(chroma_collection: Collection, question:str) -> dict:
    temp_query_results = chroma_collection.query(
        query_texts=[question],
        n_results=TOP_N
    )
    return [doc for doc, dist in zip(temp_query_results["documents"][0], temp_query_results["distances"][0]) if dist <= CHUNK_THRESHOLD]

def query_document(chroma_collection: Collection, chunks: list, document_path: Path):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context_window_limit = 200000
    context_threshold = 0.85
    model = "claude-haiku-4-5"
    while (True):
        question = input("Please ask your question: ")
        if (question == "quit"):
            print("Closing session")
            break
        if (not question):
            print("No message received")
            continue

        relevant_chunks = "\n\n".join(find_relevant_chunks(chroma_collection, question))

        if not relevant_chunks:
            print("No relevant context can be found, please try again")
            continue

        user_input = f"Here is the relevant context from the document: {relevant_chunks} \n\n Question: {question}"

        message = [{
                "content": user_input,
                "role": "user"
                }]
        
        token_count = client.messages.count_tokens(model=model, messages=message).input_tokens
        if (token_count > context_window_limit):
            print("This request will surpass the context window limit and will not be performed. Please retry this request or start a new session")
            continue
        elif (token_count > context_window_limit * context_threshold):
            print("Warning, you are nearly at this model's token limit (85 percent capacity)")

        try:
            with client.messages.stream(
                max_tokens=1024,
                messages=message,
                model=model,
                system=SYSTEM_PROMPT
            ) as stream:
                print("Claude: ", end="")
                response = ''
                for text in stream.text_stream:
                    print(text, end='', flush=True)
                    response += text
                print()    

        except anthropic.APIConnectionError as e:
            print("The server could not be reached, please try again later")
        except anthropic.RateLimitError as e:
            print("Too many requests or tokens used, try waiting up to a minute before sending in more requests")
        except anthropic.APIStatusError as e:
            print("Request unsuccessful, please try again later")

def hash_document_path(document_path: Path) -> str:
    return hashlib.sha256(str(document_path).encode()).hexdigest()

def compute_embeddings(chroma_collection: Collection, chunks:list, document_path: str):
    chunk_ids = [hash_document_path(document_path) + f"{i}" for i in range(len(chunks))]

    chroma_collection.upsert(
        ids=chunk_ids,
        documents=chunks,
    )

def main():
    # Get document path and get document
    path = input("Please input the document file path: ")
    print("Loading model and processing document...")
    document_path = Path(path)
    chunks = doc_to_chunks(document_path)
    chroma_client = chromadb.PersistentClient(path="./chromadb") # might need to be a global variable
    chroma_collection = chroma_client.get_or_create_collection(name=document_path.name.lower())
    compute_embeddings(chroma_collection, chunks, document_path)
    query_document(chroma_collection, chunks, document_path)

if __name__ == "__main__":
    main()



