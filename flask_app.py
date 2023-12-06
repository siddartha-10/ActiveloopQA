from flask import Flask, request, jsonify, render_template
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.deeplake import DeepLake
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")

app = Flask(__name__)

# Load the existing DeepLake dataset
dataset_path = f"hub://siddartha10/manufacturing_CSI"
db = DeepLake(
    dataset_path=dataset_path,  # org_id stands for your username or organization from activeloop
    embedding=OpenAIEmbeddings(),
    runtime={"tensor_db": True},
    token=activeloop_key,
    # overwrite=True, # user overwrite flag if you want to overwrite the full dataset
    read_only=True,
)

# Configure the retriever with deep_memory=True
retriever = db.as_retriever()
retriever.search_kwargs["deep_memory"] = True
retriever.search_kwargs["k"] = 10

# Initialize LangChain pipeline
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
memory = ConversationBufferWindowMemory(
        memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        data = request.get_json()
        prompt = data['prompt']

        # Run LangChain pipeline to generate the answer
        result = conversation_chain.run(prompt)

        # Extract the answer from the LangChain output
        answer = result
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=3002)
