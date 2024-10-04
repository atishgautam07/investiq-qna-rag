import os
import re
import PyPDF2
import json
import time

import nltk
# nltk.download('punkt')

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables from the .envrc file
load_dotenv('../.envrc')

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

dataPath = "../data/papers"
processedPapersPath = "../data/research_papers.json"



## read pdf documents
def read_papers(papers_folder):
    papers = []
    for filename in os.listdir(papers_folder):
        if filename.endswith('.pdf'):
            # print (filename)
            filepath = os.path.join(papers_folder, filename)
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    title = reader.metadata
                    text += page.extract_text()
                papers.append({
                    'filename': filename,
                    'metadata':title,
                    'content': text
                })
    return papers

def preprocess_text(text):
    # Remove unwanted characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

## process and chunk texts
def chunk_text(text, max_tokens=700):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


## generate summary nad key points using LLM
def generate_metadata(chunk_text, source):
    if source == 'paper':
        prompt = (
            "You are an expert financial analyst and researcher specializing in trading strategies and behavioral finance. "
            "Analyze the following excerpt from a research paper and provide a very brief summary in less than 20 words and a maximum of 3-5 key topics."
            "Output the results as a dictionary in plain text without any code block formatting or escape characters, the dictionary should contain the following keys: 'summary', 'key_topics'.\n\n"

            f"Text:\n{chunk_text}\n\n"

            "Output: summary: <summary>, key_topics: <key_topics>"
        )
    response = client.chat.completions.create(
        model='gpt-4o-mini',  # Specify the model
        messages=[
            # The conversation history, starting with the user's prompt
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,        # Limit the response tokens
        temperature=0.5,       # Control the randomness
        n=1,                   # Number of responses to generate
        stop=None              # When to stop generating tokens
    )
    metadata_text = response.choices[0].message.content.strip()
    return metadata_text

## call if genMetadata == True else read from existing json
def process_metadata():
    papers_folder = dataPath
    
    print ("Reading papers.")
    papers_data = read_papers(papers_folder)
    processed_papers = []

    print ("Processing text and chunking.")
    for paper in papers_data:
        text = preprocess_text(paper['content'])
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            processed_papers.append({
                'source': 'paper',
                'chunk_id': f"{paper['filename'].split('.')[0]}_chunk_{i}",
                'content': chunk
            })

    print ("Generating metadata.")
    for item in tqdm(processed_papers):
        metadata = generate_metadata(item['content'], source='paper')
        item['metadata'] = metadata
        
    return processed_papers


def load_data(genMetadata = False):

    if genMetadata:
        processed_papers = process_metadata()
    else:
        papers_path = processedPapersPath
        with open(papers_path, 'r') as json_file:
            processed_papers = json.load(json_file)

    for item in processed_papers:
        
        metdt = item['metadata'].split('\n')

        item['summary'] = metdt[0][len("summary:"):].strip()
        item['key_topics'] = [i.strip() for i in metdt[1][len("key_topics:"):].strip().split(',')]
        del item['metadata']

    all_data = processed_papers #processed_letters #+ processed_reports
    print (f"length of chunks - {len(all_data)}")

    return all_data


if __name__ == "__main__":
    load_data()