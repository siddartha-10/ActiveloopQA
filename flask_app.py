# from flask import Flask, request, jsonify, render_template
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.vectorstores.deeplake import DeepLake
# import os
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# import json
# from dotenv import load_dotenv
# from langchain.memory import ConversationSummaryMemory

# # Load environment variables from the .env file
# load_dotenv()

# # Access the OpenAI key
# openai_key = os.getenv("OPENAI_API_KEY")
# activeloop_key = os.getenv("ACTIVELOOP_TOKEN")

# app = Flask(__name__)

# # # Load the existing DeepLake dataset
# dataset_path = f"hub://siddartha10/manufacturing_CSI_2048"
# db = DeepLake(
#     dataset_path=dataset_path,  # org_id stands for your username or organization from activeloop
#     embedding=OpenAIEmbeddings(),
#     runtime={"tensor_db": True},
#     token=activeloop_key,
#     # overwrite=True, # user overwrite flag if you want to overwrite the full dataset
#     read_only=True,
# )

# # Configure the retriever with deep_memory=True
# retriever = db.as_retriever()
# retriever.search_kwargs["deep_memory"] = True
# retriever.search_kwargs["k"] = 10


# llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
# memory = ConversationSummaryMemory(llm=llm,
#     memory_key='chat_history', return_messages=True,max_token_length=500)

# conversation_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory
# )


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/generate_answer', methods=['POST'])
# def generate_answer():
#     try:
#         data = request.get_json()
#         prompt = data['prompt']
#         print(prompt)
#         prompt1 = memory.load_memory_variables({})['chat_history'][0].content
#         result = conversation_chain.run(prompt+"\n"+prompt1)
#         # Extract the answer from the LangChain output
#         answer = result
        
#         return app.response_class(
#             response=json.dumps({"message": answer, "status": "success"}),
#             status=200,
#             mimetype='application/json'
#         )

#     except Exception as e:
#         print(e)
#         return app.response_class(
#             response=json.dumps({"message": "Error While generating answers, Please try again", "status": "error"}),
#             status=500,
#             mimetype='application/json'
#         )

# if __name__ == '__main__':
#     app.run(port=5002, debug=True)
from flask import Flask, request, jsonify, render_template
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores.deeplake import DeepLake
import os
from dotenv import load_dotenv
import json

# Load environment variables from the .env file
load_dotenv()

# Access the OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")

app = Flask(__name__)

# Load the existing DeepLake dataset
dataset_path = f"hub://siddartha10/manufacturing_CSI_2048"
db = DeepLake(
    dataset_path=dataset_path,
    embedding=OpenAIEmbeddings(),
    runtime={"tensor_db": True},
    token=activeloop_key,
    read_only=True,
)

# Configure the retriever with deep_memory=True
retriever = db.as_retriever()
retriever.search_kwargs["deep_memory"] = True
retriever.search_kwargs["k"] = 10

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
memory = ConversationSummaryMemory(llm=llm, memory_key='chat_history', return_messages=True, max_token_length=500)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
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
        print(prompt)
        
        # Get the conversation history from memory
        prompt1 = memory.load_memory_variables({})['chat_history'][0].content
        
        # Run the conversation chain
        result = conversation_chain.run(prompt + "\n" + prompt1)
        
        # Extract the answer from the LangChain output
        answer = result
        

        return app.response_class(
            response=json.dumps({"message": answer, "status": "success"}),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print(e)
        return app.response_class(
            response=json.dumps({"message": "Error While generating answers, Please try again", "status": "error"}),
            status=500,
            mimetype='application/json'
        )

if __name__ == '__main__':
    app.run(port=5002, debug=True)
