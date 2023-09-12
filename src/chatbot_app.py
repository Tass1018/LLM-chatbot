import chainlit as cl
from my_chatbot_module import \
    chatbot_response  # Replace with the actual module and function that handles chatbot responses

# Load your knowledge base (replace with the actual path to your knowledge base)
knowledge_base = load_knowledge_base('path/to/your/knowledge_base.json')


def load_knowledge_base(path):
    with open(path, 'r') as file:
        return json.load(file)


# Define a function to handle user queries and generate responses
def handle_query(query):
    response = chatbot_response(knowledge_base, query)
    return response


# Create a Chainlit app
app = cl.App()

# Create a text input field for users to enter their queries
query_input = app.text_input("Your question:", placeholder="Type your question here...")

# Create a button that users can click to submit their queries
submit_button = app.button("Submit")

# Create a text area to display the chatbot's responses
response_output = app.text_area("Chatbot's response:", height=200, disabled=True)


# Define what happens when the submit button is clicked
@submit_button.on_click
def on_submit():
    # Get the user's query from the text input field
    query = query_input.get()

    # Get the chatbot's response using the handle_query function
    response = handle_query(query)

    # Display the chatbot's response in the text area
    response_output.set(response)


# Run the Chainlit app
if __name__ == "__main__":
    app.run()
