import pymupdf
import pymupdf4llm
import pathlib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
import warnings

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

ACCEPTED_FILE_TYPES = [".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".svg", ".txt", ".doc", ".docx"]
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))
SYSTEM_PROMPT = "You are a helpful, kind, and thoughtful assistant helping the user answer questions they have about a document.\
    You should do your best to answer the questions posed to you by the user, but if you do not know the answer, likely because \
    the answer is not in the information, you should let the user know that you do not have an answer but try to help the user by\
    telling them what information would be needed to answer the question. You should not give answers that are not relevant to the\
    context received or use outside information to guess at an answer. Stick strictly to the context provided."

# Convert document to markdown format in chunks
def doc_to_chunks(path: pathlib.Path) -> list:
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

def extract_keywords(question: str) -> list:
    tokenized_words = word_tokenize(question)
    keywords = [word.lower() for word in tokenized_words if word.lower() not in STOP_WORDS]
    return keywords

def find_relevant_chunks(keywords: list, chunks: list) -> list:
    relevant_chunks = []
    for chunk in chunks:
        chunk_lowercase = chunk.lower()
        for keyword in keywords:
            if keyword in chunk_lowercase:
                relevant_chunks.append(chunk)
                break
    return relevant_chunks

def query_document(chunks: list):
    load_dotenv()

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

        words = extract_keywords(question)
        relevant_chunks = "\n\n".join(find_relevant_chunks(words, chunks))
        
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


# Get document path and get document
document_path = input("Please input the document file path: ")
path = pathlib.Path(document_path)
chunks = doc_to_chunks(path)
query_document(chunks)



