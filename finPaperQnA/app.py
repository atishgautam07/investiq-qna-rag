import streamlit as st
import rag
from db_utils import init_db, save_conv, save_feedback  # Import the database functions

init_db()

# Initialize session state variables
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''
if 'question' not in st.session_state:
    st.session_state['question'] = ''
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = None

def main():
    st.title("InvestIQ - Financial Research Q&A Application")

    # Input box for the question
    st.session_state['question'] = st.text_input("Enter your question:")

    # Generate button
    if st.button("Generate"):
        if st.session_state['question']:
            # Call the RAG function
            result = rag.rag_pipeline(st.session_state['question'])
            answer = result[0]
            prompt = result[1]
            response_time = result[2]
            prompt_tokens = result[3]
            completion_tokens = result[4]
            total_tokens = result[5]
            total_cost = result[6]

            st.session_state['answer'] = answer
            st.session_state['feedback'] = None  # Reset feedback when a new question is asked

            # Save the conversation to the database
            conversation_id = save_conv(
                question=st.session_state['question'],
                answer=answer,
                model_used='gpt-4o-mini',
                response_time=response_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost=total_cost
            )
            st.session_state['conversation_id'] = conversation_id
        else:
            st.warning("Please enter a question.")

    # Display the answer
    if st.session_state['answer']:
        st.write("**Answer:**")
        st.write(st.session_state['answer'])

        # Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key="positive_feedback"):
                if st.session_state['feedback'] is None:
                    st.session_state['feedback'] = 1
                    save_feedback(st.session_state['conversation_id'], 1)
                    st.success("Thank you for your feedback!")
                else:
                    st.info("You have already provided feedback.")
        with col2:
            if st.button("👎", key="negative_feedback"):
                if st.session_state['feedback'] is None:
                    st.session_state['feedback'] = -1
                    save_feedback(st.session_state['conversation_id'], -1)
                    st.success("Thank you for your feedback!")
                else:
                    st.info("You have already provided feedback.")


if __name__ == '__main__':
    main()
