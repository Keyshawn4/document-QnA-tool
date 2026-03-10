import pymupdf
import pymupdf4llm
import pathlib
from pathlib import Path
import os
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
from operator import itemgetter
import hashlib
import numpy as np
import json
from transformers import logging as transformers_logging
from transformers.utils import logging

load_dotenv()

logging.disable_progress_bar()
logging.get_logger("transformers").setLevel(logging.ERROR)
logging.get_logger("huggingface_hub").setLevel(logging.ERROR)

ACCEPTED_FILE_TYPES = [".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".svg", ".txt", ".doc", ".docx"]
SYSTEM_PROMPT = "You are a helpful, kind, and thoughtful assistant helping the user answer questions they have about a document.\
    You should do your best to answer the questions posed to you by the user, but if you do not know the answer, likely because \
    the answer is not in the information, you should let the user know that you do not have an answer but try to help the user by\
    telling them what information would be needed to answer the question. You should not give answers that are not relevant to the\
    context received or use outside information to guess at an answer. Stick strictly to the context provided."
CHUNK_THRESHOLD = 0.25
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


def find_relevant_chunks(transformer_model, question: str, chunk_data: list) -> list:
    relevant_chunks = []
    threshold_chunks = [] # for chunks that have passed the minimum threshold
    question_embedding = transformer_model.encode(question, show_progress_bar=False)
    for chunk in chunk_data:
        similarity_score = transformer_model.similarity(question_embedding, chunk[1])
        if similarity_score >= CHUNK_THRESHOLD:
            threshold_chunks.append((chunk[0], chunk[1], similarity_score))

    threshold_chunks = sorted(threshold_chunks, key=itemgetter(2), reverse=True)
    for i in range(TOP_N):
        if i < len(threshold_chunks):
            relevant_chunks.append(threshold_chunks[i][0])
        else:
            break
    
    return relevant_chunks

def query_document(transformer_model, chunks: list, document_path: Path):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context_window_limit = 200000
    context_threshold = 0.85
    model = "claude-haiku-4-5"
    chunk_data = [(chunk, transformer_model.encode(chunk, show_progress_bar=False)) for chunk in chunks]
    while (True):
        question = input("Please ask your question: ")
        if (question == "quit"):
            print("Closing session")
            break
        if (not question):
            print("No message received")
            continue

        chunk_text_embeddings = compute_embeddings(transformer_model, question, document_path, chunk_data)
        relevant_chunks = "\n\n".join(find_relevant_chunks(transformer_model, question, chunk_text_embeddings))

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

def compute_embeddings(transformer_model, question: str, document_path: Path, chunk_data: list):
    target_directory = Path("embeddings_cache")
    if not target_directory.is_dir():
        target_directory.mkdir(parents=True, exist_ok=True)
    document_hash = hash_document_path(document_path)
    cache_path = pathlib.Path(target_directory / document_hash)
    embeddings_file_path = cache_path.with_suffix('.npy')
    chunk_file_path = cache_path.with_suffix('.json')

    if not embeddings_file_path.exists():
        chunk_text, chunk_embeddings = zip(*chunk_data)
        with chunk_file_path.open('w') as file:
            json.dump(chunk_text, file)
        np.save(embeddings_file_path, chunk_embeddings)
        return chunk_data
    else:
        with open(chunk_file_path,'r') as file:
            chunk_text = json.load(file)
        chunk_embeddings = np.load(embeddings_file_path)
        chunk_data = zip(chunk_text, chunk_embeddings)
        return chunk_data

def main():
    # Get document path and get document
    path = input("Please input the document file path: ")
    print("Loading model and processing document...")
    document_path = Path(path)
    chunks = doc_to_chunks(document_path)
    from sentence_transformers import SentenceTransformer # This import statement happens late because it slows down the program so it happens right before it is necessary
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_document(transformer_model, chunks, document_path)

if __name__ == "__main__":
    main()



