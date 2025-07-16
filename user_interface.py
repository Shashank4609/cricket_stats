import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq.chat_models import ChatGroq
import traceback
from datetime import datetime
from database import submit_feedback, get_db_connection, find_similar_queries
from prompt import SYSTEM_PROMPT  # Make sure this exists

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="DataBot", layout="wide")

# CSS for styling
st.markdown("""
<style>
.chat-box {padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1f77b4;}
.user-box {background-color: #f0f2f6; border-left-color: #ff7f0e;}
.assistant-box {background-color: #e8f4fd;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False


# Load dataset
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# LLM init
def initialize_llm():
    api_key = os.getenv("api_key")
    if not api_key:
        st.error("API key missing in .env")
        return None
    try:
        return ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")
    except Exception as e:
        st.error(f"LLM init failed: {e}")
        return None


# Prompt builder
def generate_prompt_code(user_query):
    return f"{SYSTEM_PROMPT}\nUser: {user_query}\nAnswer:"


# Safe code executor
def execute_code(code, df):
    try:
        safe_env = {
            'df': df, 'pd': pd, 'len': len, 'sum': sum,
            'max': max, 'min': min, 'sorted': sorted
        }
        return eval(code, {"__builtins__": {}}, safe_env), None
    except Exception as e:
        return None, f"{e}\n{traceback.format_exc()}"


# Load model & dataset
df = load_data("final_dataset.csv") if os.path.exists("final_dataset.csv") else None
llm = initialize_llm()

if df is None or llm is None:
    st.warning("Dataset not loaded or API key missing.")
else:
    st.title("📊 DataBot: Ask IPL Questions")

    # Add a sidebar for similar queries feature
    with st.sidebar:
        st.header("🔍 Similar Queries")
        if st.button("Find Similar Queries"):
            if 'user_query' in st.session_state:
                similar_queries = find_similar_queries(st.session_state['user_query'], limit=3)
                if similar_queries:
                    st.subheader("Similar past queries:")
                    for i, (conv_id, query, response, code, rating, similarity) in enumerate(similar_queries):
                        with st.expander(f"Query {i + 1} (Similarity: {similarity:.2f})"):
                            st.write(f"**Query:** {query}")
                            st.write(f"**Rating:** {rating}")
                            st.code(code, language='python')
                else:
                    st.info("No similar queries found yet.")

    user_input = st.text_input("Enter your question")

    if st.button("Submit") and user_input.strip():
        # Reset feedback state for new query
        st.session_state.feedback_submitted = False
        st.session_state.show_feedback = False

        with st.spinner("Generating response..."):
            prompt = generate_prompt_code(user_input)

            try:
                response = llm.invoke(prompt)
                generated_code = response.content.strip()

                # Execute code
                result, error = execute_code(generated_code, df)

                # Save to session_state BEFORE showing results
                st.session_state["user_query"] = user_input
                st.session_state["model_response"] = str(result) if result else "No result generated."
                st.session_state["pandas_code"] = generated_code
                st.session_state["conversation_id"] = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.show_feedback = True

                # Store every interaction in database (even without feedback)
                try:
                    submit_feedback(
                        st.session_state["conversation_id"],
                        st.session_state["user_query"],
                        st.session_state["model_response"],
                        st.session_state["pandas_code"],
                        "no_feedback",  # Default status for automatic logging
                        None
                    )
                except Exception as e:
                    st.error(f"Error logging interaction: {e}")

                # Show assistant's response
                st.markdown("### 🧠 Generated Code")
                st.code(generated_code, language='python')

                st.markdown("### 📊 Result")
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)
                if error:
                    st.error(f"Error:\n{error}")

            except Exception as e:
                st.error(f"LLM error: {e}")
                st.session_state.show_feedback = False

    # Feedback section - only show if we have a response and haven't submitted feedback yet
    if st.session_state.get('show_feedback', False) and not st.session_state.get('feedback_submitted', False):
        st.markdown("---")
        st.markdown("### 💬 Was this answer helpful?")

        # Use columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes, it was helpful!", key="positive_feedback"):
                try:
                    # Update the existing record with positive feedback
                    submit_feedback(
                        st.session_state.get("conversation_id", ""),
                        st.session_state.get("user_query", ""),
                        st.session_state.get("model_response", ""),
                        st.session_state.get("pandas_code", ""),
                        "up",
                        "User found the response helpful"
                    )
                    st.success("✅ Thanks for your positive feedback!")
                    st.session_state.feedback_submitted = True
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")

        with col2:
            if st.button("👎 No, needs improvement", key="negative_feedback"):
                st.session_state.show_comment_box = True

        # Show comment box if negative feedback was clicked
        if st.session_state.get('show_comment_box', False) and not st.session_state.get('feedback_submitted', False):
            st.markdown("**Please tell us what went wrong:**")
            comment = st.text_area("Your feedback helps us improve", key="feedback_comment")

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit Feedback", key="submit_negative_feedback"):
                    if comment.strip():
                        try:
                            # Submit negative feedback with comment
                            submit_feedback(
                                st.session_state.get("conversation_id", ""),
                                st.session_state.get("user_query", ""),
                                st.session_state.get("model_response", ""),
                                st.session_state.get("pandas_code", ""),
                                "down",
                                comment
                            )
                            st.success("✅ Thanks for helping us improve!")
                            st.session_state.feedback_submitted = True
                            st.session_state.show_comment_box = False
                        except Exception as e:
                            st.error(f"Error submitting feedback: {e}")
                    else:
                        st.warning("Please provide some feedback before submitting.")

            with col2:
                if st.button("Cancel", key="cancel_feedback"):
                    st.session_state.show_comment_box = False