import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elastic_transport import ConnectionError
from sentence_transformers import SentenceTransformer

import ingest

# Load environment variables from the .envrc file
load_dotenv('../.envrc')

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

index_name = 'finance_paper_qna'
model = SentenceTransformer('all-MiniLM-L6-v2')  #('FinLang/finance-embeddings-investopedia') #('all-MiniLM-L6-v2')  #("philschmid/bge-base-financial-matryoshka")

# Create an Elasticsearch client instance
def get_elasticsearch_client(retries=5, delay=5):
    es_host = os.getenv('ELASTICSEARCH_HOST')
    es_port = int(os.getenv('ELASTICSEARCH_PORT'))

    for attempt in range(retries):
        try:
            es = Elasticsearch([{'scheme': 'http','host': es_host, 'port': es_port}])
            # Test the connection
            if es.ping():
                print("Connected to Elasticsearch")
                return es
            else:
                raise ConnectionError("Elasticsearch ping failed.")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise ConnectionError("Could not connect to Elasticsearch after several attempts.")

# Load data (if you haven't indexed already)
all_data = ingest.load_data()

# Use the function to get the Elasticsearch client
es = get_elasticsearch_client()

# Function to setup Elasticsearch index and bulk insert data
def setup_elasticsearch_index(es, index_name, model, data):
    index_mapping = {
        'mappings': {
            'properties': {
                'source': {'type': 'keyword'},
                'chunk_id': {'type': 'keyword'},
                'content': {'type': 'text'},
                'summary': {'type': 'text'},
                'key_topics': {'type': 'text'},
                'embedding': {'type': 'dense_vector', 'dims': 384}
            }
        }
    }

    # Create or update index
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)
        bulk(es, generate_actions(model, data, index_name))
        print(f"Data indexed successfully in {index_name}")
    else:
        print(f"Index {index_name} already exists. Skipping indexing.")

# Function to generate actions for bulk indexing
def generate_actions(model, data, index_name):
    for item in data:
        embedding = model.encode(item['content']).tolist()
        yield {
            '_index': index_name,
            '_id': item['chunk_id'],
            '_source': {
                'source': item['source'],
                'chunk_id': item['chunk_id'],
                'content': item['content'],
                'summary': item.get('summary', ''),
                'key_topics': item.get('key_topics', ''),
                'embedding': embedding
            }
        }

setup_elasticsearch_index(es, index_name, model, all_data)


# index_mapping = {
#     'mappings': {
#         'properties': {
#             'source': {'type': 'keyword'},
#             'chunk_id': {'type': 'keyword'},
#             'content': {'type': 'text'},
#             'summary': {'type': 'text'},
#             'key_topics': {'type': 'text'},
#             'embedding': {
#                 'type': 'dense_vector',
#                 'dims': 384  # Adjust based on your embedding dimensions
#             }
#         }
#     }
# }

# # Create or update the index
# print (index_name)
# if not es.indices.exists(index=index_name):
#     es.indices.create(index=index_name, body=index_mapping)
# else:
#     es.indices.put_mapping(index=index_name, body=index_mapping)


# def generate_actions(data):
#     for item in data:
#         embedding = model.encode(item['content']).tolist()  #get_embeddings(item['content']).tolist() #
#         yield {
#             '_index': index_name,
#             '_id': item['chunk_id'],
#             '_source': {
#                 'source': item['source'],
#                 'chunk_id': item['chunk_id'],
#                 'content': item['content'],
#                 'summary': item.get('summary', ''),
#                 'key_topics': item.get('key_topics', ''),
#                 'embedding': embedding
#             }
#         }

# # Bulk index the data
# bulk(es, generate_actions(all_data))


def hybrid_search(query, index=index_name, keyword_top_k=10, final_top_k=10):
    """
    Performs a hybrid search using keyword (multi_match) and semantic (k-NN) search, then re-ranks the results.

    Parameters:
    - query (str): The user's question.
    - index (str): The Elasticsearch index name.
    - keyword_top_k (int): Number of top documents to retrieve from keyword search.
    - final_top_k (int): Number of top documents to return after re-ranking.

    Returns:
    - List of dictionaries containing the re-ranked search results.
    """

    # Step 1: Perform Keyword Search (multi_match)
    keyword_query = {
        'size': keyword_top_k,
        'query': {
            'multi_match': {
                'query': query,
                'fields': ['content','summary^2','key_topics^2'],
                'fuzziness': 'AUTO'
            }
        },
        '_source': ['source', 'chunk_id', 'content', 'embedding', 'summary', 'key_topics']
    }
    keyword_response = es.search(index=index, body=keyword_query)
    keyword_hits = keyword_response['hits']['hits']

    # Step 2: Perform Semantic Search (k-NN)
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # Build the semantic search query
    # Semantic search using script_score
    semantic_query = {
        'size': keyword_top_k,
        'query': {
            'script_score': {
                'query': {'match_all': {}},
                'script': {
                    'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    'params': {'query_vector': query_embedding}
                }
            }
        },
        '_source': ['source', 'chunk_id', 'content', 'embedding', 'summary', 'key_topics']
    }

    semantic_response = es.search(index=index, body=semantic_query)
    semantic_hits = semantic_response['hits']['hits']

    # Step 3: Combine Results
    combined_hits = {}
    # Process keyword search results
    for hit in keyword_hits:
        chunk_id = hit['_id']
        combined_hits[chunk_id] = {
            'source': hit['_source']['source'],
            'year': hit['_source'].get('year', ''),
            'chunk_id': chunk_id,
            'content': hit['_source']['content'],
            'summary': hit['_source']['summary'],
            'key_topics': hit['_source']['key_topics'],
            'keyword_score': hit['_score'],
            'semantic_score': 0  # Will be updated if exists in semantic_hits
        }
    # Process semantic search results
    for hit in semantic_hits:
        chunk_id = hit['_id']
        if chunk_id in combined_hits:
            combined_hits[chunk_id]['semantic_score'] = hit['_score']
        else:
            combined_hits[chunk_id] = {
                'source': hit['_source']['source'],
                # 'year': hit['_source'].get('year', ''),
                'chunk_id': chunk_id,
                'content': hit['_source']['content'],
                'summary': hit['_source']['summary'],
                'key_topics': hit['_source']['key_topics'],
                'keyword_score': 0,
                'semantic_score': hit['_score']
            }

    # Step 4: Re-rank the Combined Results
    # Normalize scores
    max_keyword_score = max(hit['keyword_score'] for hit in combined_hits.values()) or 1
    max_semantic_score = max(hit['semantic_score'] for hit in combined_hits.values()) or 1

    for hit in combined_hits.values():
        hit['keyword_score_normalized'] = hit['keyword_score'] / max_keyword_score
        hit['semantic_score_normalized'] = hit['semantic_score'] / max_semantic_score
        # Combine scores with weights (adjust weights as needed)
        hit['combined_score'] = (0.6 * hit['keyword_score_normalized']) + (0.4 * hit['semantic_score_normalized'])

    # Sort the hits based on combined score
    re_ranked_hits = sorted(combined_hits.values(), key=lambda x: x['combined_score'], reverse=True)

    # Step 5: Return Top-K Re-Ranked Documents
    final_results = re_ranked_hits[:final_top_k]
    return final_results


def build_prompt(query, search_results):
    """
    Builds a prompt for the LLM using the query and search results.

    Parameters:
    - query (str): The user's question or query.
    - search_results (list): List of retrieved documents.
    
    Returns:
    - The formatted prompt string.
    """
    # Instruction to the LLM
    instruction = (
        "You are a financial analyst assistant with deep knowledge of trading strategies and behavioral finance. "
        "Using the provided context, answer the user's question. "
        "Use only the facts from the context when answering the question."
        "If the context is insufficient, let the user know. "
        "Provide clear, concise explanations, and include relevant insights from the research papers.\n\n"
    )

    # Build context from search results
    context = ""
    for result in search_results:
        # source = result['source']
        # year = result.get('year', '')
        content = result['content']
        summary = result['summary']
        key_topics = result['key_topics']
        context += f"Content: {content}\n Summary: {summary}\n Key topics: {key_topics}\n\n"  #


    # Assemble the prompt
    prompt = f"{instruction}Context:\n{context}\nQuestion: {query}\nAnswer:"
    return prompt

def call_llm(prompt, model='gpt-4o-mini'):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=2000,
        temperature=0.25
    )
    end_time = time.time()
    answer = response.choices[0].message.content.strip()
    response_time = end_time - start_time

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return answer, token_stats, response_time

def calculate_cost(model, total_tokens):
    """
    Calculates the cost of the API call based on the model and total tokens.

    Parameters:
    - model (str): The name of the model used.
    - total_tokens (int): The total number of tokens used.

    Returns:
    - total_cost (float): The cost in USD.
    """
    if model == 'gpt-3.5-turbo':
        cost_per_1k_tokens = 0.002  # USD per 1K tokens
    elif model == 'gpt-4':
        cost_per_1k_tokens = 0.06  # Adjust based on actual pricing
    else:
        cost_per_1k_tokens = 0.002  # Default to gpt-3.5-turbo pricing

    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return total_cost


def rag_pipeline(query):
    """
    Runs the Retrieval-Augmented Generation pipeline for a given query.

    Parameters:
    - query (str): The user's question or query.
    
    Returns:
    - The final answer from the LLM.
    """


    print ("Searching.")
    search_results = hybrid_search(query)
    print ("Building prompt.")
    prompt = build_prompt(query, search_results)

    # Step 3: Call LLM
    try:
        answer, token_stats, response_time = call_llm(prompt)
        # Extract usage data
        prompt_tokens = token_stats['prompt_tokens']
        completion_tokens = token_stats['completion_tokens']
        total_tokens = token_stats['total_tokens']
        # Calculate cost based on model and total tokens
        total_cost = calculate_cost(model='gpt-4o-mini', total_tokens=total_tokens)
    except Exception as e:
        answer = f"An error occurred while generating the answer: {e}"
        prompt_tokens = completion_tokens = total_tokens = total_cost = response_time = None

    return answer, prompt, response_time, prompt_tokens, completion_tokens, total_tokens, total_cost


# Main entry point for querying
if __name__ == "__main__":
    
    # Call RAG pipeline with a query
    query = "Mention 3 efficient trading strategies from the knowledge base."
    answer = rag_pipeline(query, es, model)
    
    # Print the final answer
    print("Answer:", answer)