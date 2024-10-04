import psycopg2
from dotenv import load_dotenv
import os
# Load environment variables from the .envrc file
load_dotenv('../.envrc')


def get_db_connection():
    """
    Returns a connection to the PostgreSQL database using environment variables.

    Environment Variables:
    - DB_HOST: Database host
    - DB_PORT: Database port
    - DB_NAME: Database name
    - DB_USER: Database user
    - DB_PASSWORD: Database password

    Returns:
    - conn: psycopg2 connection object
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None



def init_db():
    """
    Initializes the database by creating the required tables if they do not exist.
    Tables:
    - conversations
    - feedback
    """
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return

    try:
        cursor = conn.cursor()

        # Create conversations table
        create_conversations_table = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            model_used VARCHAR(100),
            response_time FLOAT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            total_cost FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create feedback table
        create_feedback_table = """
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversations(id),
            feedback INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(create_conversations_table)
        cursor.execute(create_feedback_table)
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing the database: {e}")
        conn.rollback()
        cursor.close()
        conn.close()


def save_conv(question, answer, model_used, response_time, prompt_tokens, completion_tokens, total_tokens, total_cost):
    """
    Saves a conversation record in the conversations table.

    Parameters:
    - question (str)
    - answer (str)
    - model_used (str)
    - response_time (float)
    - prompt_tokens (int)
    - completion_tokens (int)
    - total_tokens (int)
    - total_cost (float)

    Returns:
    - conversation_id (int): The unique ID of the inserted conversation record.
    """
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return None

    try:
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO conversations (question, answer, model_used, response_time, prompt_tokens, completion_tokens, total_tokens, total_cost)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """

        cursor.execute(insert_query, (question, answer, model_used, response_time, prompt_tokens, completion_tokens, total_tokens, total_cost))
        conversation_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return conversation_id
    except Exception as e:
        print(f"Error saving conversation: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return None



def save_feedback(conversation_id, feedback_score):
    """
    Saves user feedback for a conversation.

    Parameters:
    - conversation_id (int): The unique ID of the conversation.
    - feedback_score (int): +1 for positive feedback, -1 for negative feedback.

    Returns:
    - None
    """
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return

    try:
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO feedback (conversation_id, feedback)
        VALUES (%s, %s);
        """

        cursor.execute(insert_query, (conversation_id, feedback_score))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error saving feedback: {e}")
        conn.rollback()
        cursor.close()
        conn.close()



# cursor.execute("DROP TABLE IF EXISTS embeddings")
# # Create table
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS embeddings (
#         chunk_id TEXT PRIMARY KEY,
#         source TEXT,
#         year TEXT,
#         summary TEXT,
#         keytopics TEXT[],
#         embedding FLOAT[]
#     );
# """)
# conn.commit()


# # Insert embeddings
# for item in all_data:
#     cursor.execute("""
#         INSERT INTO embeddings (chunk_id, source, year, summary, keytopics, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s)
#         ON CONFLICT (chunk_id) DO NOTHING;
#     """, (
#         item['chunk_id'],
#         item['source'],
#         item['year'],
#         item['summary'],
#         item['key_topics'],
#         item['embedding'].tolist(),  # Convert numpy array to list
#     ))
# conn.commit()
# conn.close()
