import psycopg2
import streamlit as st
import traceback
import numpy as np
import torch

# Set this before importing sentence_transformers
torch.set_num_threads(1)
# from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import os


# Initialize embedding model (cached)
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None


def get_embeddings(text, model):
    """Generate embeddings for given text"""
    try:
        if model is None:
            return None
        embedding = model.encode([text])
        return embedding[0].tolist()  # Convert to list for PostgreSQL
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="iplchatbot",
            user="postgres",
            password="abcd1234",
            host="localhost",
            port="5432"
        )
        conn.cursor().execute("SELECT 1")  # Test connection
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def check_table_structure():
    """Check if the table has the required vector columns"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        cursor = conn.cursor()

        # Check if vector columns exist
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'user_feedback' 
            AND column_name IN ('query_embedding', 'response_embedding', 'code_embedding')
        """)

        existing_columns = cursor.fetchall()
        vector_columns = [col[0] for col in existing_columns]

        print(f"Existing vector columns: {vector_columns}")

        return len(vector_columns) == 3

    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def migrate_table_to_vector():
    """Migrate existing table to include vector columns"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        cursor = conn.cursor()

        # First, enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Check if vector columns already exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'user_feedback' 
            AND column_name IN ('query_embedding', 'response_embedding', 'code_embedding')
        """)

        existing_columns = [row[0] for row in cursor.fetchall()]

        # Add missing vector columns
        if 'query_embedding' not in existing_columns:
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN query_embedding vector(384)")
            print("✅ Added query_embedding column")

        if 'response_embedding' not in existing_columns:
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN response_embedding vector(384)")
            print("✅ Added response_embedding column")

        if 'code_embedding' not in existing_columns:
            cursor.execute("ALTER TABLE user_feedback ADD COLUMN code_embedding vector(384)")
            print("✅ Added code_embedding column")

        conn.commit()

        # Create indexes for vector similarity search
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS query_embedding_idx 
                ON user_feedback USING ivfflat (query_embedding vector_cosine_ops)
            """)
        except Exception as e:
            print(f"Note: Could not create query_embedding index: {e}")

        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS response_embedding_idx 
                ON user_feedback USING ivfflat (response_embedding vector_cosine_ops)
            """)
        except Exception as e:
            print(f"Note: Could not create response_embedding index: {e}")

        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS code_embedding_idx 
                ON user_feedback USING ivfflat (code_embedding vector_cosine_ops)
            """)
        except Exception as e:
            print(f"Note: Could not create code_embedding index: {e}")

        conn.commit()
        print("✅ Table migration completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error migrating table: {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def create_feedback_table():
    """Create the feedback table with pgvector extension if it doesn't exist"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        cursor = conn.cursor()

        # Enable pgvector extension first
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create table with vector columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(255) UNIQUE,
                user_query TEXT,
                model_response TEXT,
                pandas_code TEXT,
                rating VARCHAR(50),
                user_comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

        # Check if we need to migrate the table to add vector columns
        if not check_table_structure():
            print("Vector columns missing. Migrating table...")
            if not migrate_table_to_vector():
                print("❌ Failed to migrate table")
                return False
        else:
            print("✅ Table structure is correct")

        print("✅ Feedback table with pgvector created/verified successfully")
        return True

    except Exception as e:
        print(f"❌ Error creating feedback table: {e}")
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def submit_feedback(conversation_id, user_query, model_response, pandas_code, rating, user_comment):
    """
    Submit feedback to PostgreSQL database with embeddings
    This function handles both automatic logging and user feedback
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Could not establish database connection")

        cursor = conn.cursor()

        # Generate embeddings
        embedding_model = load_embedding_model()
        query_embedding = get_embeddings(user_query, embedding_model) if user_query else None
        response_embedding = get_embeddings(model_response, embedding_model) if model_response else None
        code_embedding = get_embeddings(pandas_code, embedding_model) if pandas_code else None

        # Check if this conversation_id already exists
        cursor.execute("""
            SELECT id FROM user_feedback WHERE conversation_id = %s
        """, (conversation_id,))

        existing_record = cursor.fetchone()

        if existing_record and rating != "no_feedback":
            # Update existing record with actual feedback
            cursor.execute("""
                UPDATE user_feedback 
                SET rating = %s, user_comment = %s, timestamp = NOW(),
                    query_embedding = %s, response_embedding = %s, code_embedding = %s
                WHERE conversation_id = %s
            """, (rating, user_comment, query_embedding, response_embedding, code_embedding, conversation_id))
            print(f"✅ Updated existing record for conversation: {conversation_id}")
        else:
            # Insert new record (for automatic logging or if no existing record)
            cursor.execute("""
                INSERT INTO user_feedback (
                    conversation_id, user_query, model_response, pandas_code, rating, user_comment, 
                    query_embedding, response_embedding, code_embedding, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                conversation_id,
                user_query,
                model_response,
                pandas_code,
                rating,
                user_comment,
                query_embedding,
                response_embedding,
                code_embedding
            ))
            print(f"✅ New record inserted for conversation: {conversation_id}")

        conn.commit()

    except Exception as e:
        print("❌ DB Error:", e)
        traceback.print_exc()
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_all_feedback():
    """Retrieve all feedback from the database"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []

        cursor = conn.cursor()

        cursor.execute("""
            SELECT conversation_id, user_query, model_response, pandas_code, 
                   rating, user_comment, query_embedding, response_embedding, 
                   code_embedding, timestamp 
            FROM user_feedback 
            ORDER BY timestamp DESC
        """)

        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"❌ Error retrieving feedback: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_feedback_stats():
    """Get statistics about feedback"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return {}

        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                rating,
                COUNT(*) as count
            FROM user_feedback 
            GROUP BY rating
        """)

        results = cursor.fetchall()
        return {row[0]: row[1] for row in results}

    except Exception as e:
        print(f"❌ Error getting feedback stats: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def find_similar_queries(query_text, limit=5):
    """Find similar queries using vector similarity search"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []

        cursor = conn.cursor()

        # Generate embedding for the input query
        embedding_model = load_embedding_model()
        query_embedding = get_embeddings(query_text, embedding_model)

        if query_embedding is None:
            return []

        # Find similar queries using cosine similarity
        cursor.execute("""
            SELECT conversation_id, user_query, model_response, pandas_code, rating,
                   1 - (query_embedding <=> %s::vector) as similarity
            FROM user_feedback 
            WHERE query_embedding IS NOT NULL
            ORDER BY query_embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))

        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"❌ Error finding similar queries: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def find_similar_responses(response_text, limit=5):
    """Find similar responses using vector similarity search"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []

        cursor = conn.cursor()

        # Generate embedding for the input response
        embedding_model = load_embedding_model()
        response_embedding = get_embeddings(response_text, embedding_model)

        if response_embedding is None:
            return []

        # Find similar responses using cosine similarity
        cursor.execute("""
            SELECT conversation_id, user_query, model_response, pandas_code, rating,
                   1 - (response_embedding <=> %s::vector) as similarity
            FROM user_feedback 
            WHERE response_embedding IS NOT NULL
            ORDER BY response_embedding <=> %s::vector
            LIMIT %s
        """, (response_embedding, response_embedding, limit))

        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"❌ Error finding similar responses: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def find_similar_code(code_text, limit=5):
    """Find similar code using vector similarity search"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []

        cursor = conn.cursor()

        # Generate embedding for the input code
        embedding_model = load_embedding_model()
        code_embedding = get_embeddings(code_text, embedding_model)

        if code_embedding is None:
            return []

        # Find similar code using cosine similarity
        cursor.execute("""
            SELECT conversation_id, user_query, model_response, pandas_code, rating,
                   1 - (code_embedding <=> %s::vector) as similarity
            FROM user_feedback 
            WHERE code_embedding IS NOT NULL
            ORDER BY code_embedding <=> %s::vector
            LIMIT %s
        """, (code_embedding, code_embedding, limit))

        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"❌ Error finding similar code: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_embedding_stats():
    """Get statistics about stored embeddings"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return {}

        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(query_embedding) as query_embeddings,
                COUNT(response_embedding) as response_embeddings,
                COUNT(code_embedding) as code_embeddings
            FROM user_feedback
        """)

        result = cursor.fetchone()
        return {
            'total_records': result[0],
            'query_embeddings': result[1],
            'response_embeddings': result[2],
            'code_embeddings': result[3]
        }

    except Exception as e:
        print(f"❌ Error getting embedding stats: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def reset_and_recreate_table():
    """Drop and recreate the table with proper vector support - USE WITH CAUTION"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        cursor = conn.cursor()

        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Drop existing table
        cursor.execute("DROP TABLE IF EXISTS user_feedback CASCADE")

        # Create new table with vector columns
        cursor.execute("""
            CREATE TABLE user_feedback (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(255) UNIQUE,
                user_query TEXT,
                model_response TEXT,
                pandas_code TEXT,
                rating VARCHAR(50),
                user_comment TEXT,
                query_embedding vector(384),
                response_embedding vector(384),
                code_embedding vector(384),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        print("✅ Table recreated successfully with vector support")
        return True

    except Exception as e:
        print(f"❌ Error recreating table: {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Create table on import
create_feedback_table()