from rag import ask_question
from utility import *
import streamlit as st
import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
def main():
    # Title for the app
    st.title('Query for Youtube Session')

    # Input for query
    query = st.text_input("Enter your query:")

    # When the button is clicked
    if st.button("Get Answer"):
        if query:
            # Call the function to get the result
            answer = ask_question(query)
            # Display the result
            st.write("Answer: ", answer["Response"])
        else:
            st.warning("Please enter a query.")

# Run the app
if __name__ == "__main__":
    main()