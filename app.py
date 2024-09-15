import json
import random
from pathlib import Path

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

from logic import generate_prompt
from resource_handler import get_all_chunks, read_vector_index

test_questions = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
with Path(__file__).parents[0].joinpath("config.json").open("r") as json_file:
    configurations = json.load(json_file)
all_chunks = get_all_chunks(configurations["chunk_file_path"])
vector_index = read_vector_index(configurations["faiss_index_file_path"])
encoder = SentenceTransformer(configurations["sentence_encoder_model"])
tokenizer = AutoTokenizer.from_pretrained(configurations["response_generator_model"])
pipe = pipeline("text-generation", model=configurations["response_generator_model"], device_map="auto",
                num_return_sequences=1, do_sample=True, top_k=5,
                max_new_tokens=50, return_full_text=False, stream=True)


def get_random_question() -> str:
    """
    Return a random question from the rag-mini-wikipedia test dataset

    :return: a random question
    """
    return random.choice(test_questions['question'].tolist())


def submit_question(user_question: str):
    """
    Generate a prompt based on the given question

    :param user_question: The input question
    :return: The generated prompt
    """
    prompt = generate_prompt(user_question, encoder, vector_index, all_chunks, tokenizer)
    return prompt


def generate_response(prompt: str):
    """
    Generate a response from the LLM and yield it word by word

    :param prompt: The input prompt
    :yield: Words from the generated response
    """
    response = pipe(prompt)
    response_text = ""
    for token in response['generated_text']:
        response_text += token
        yield response_text


with gr.Blocks(theme="gradio/monochrome", title="RAG Project Application") as rag_app:
    gr.Markdown("# 🚀 RAG Project Application")
    gr.Markdown("## 📚 Introduction")
    gr.Markdown("RAG, or **R**etrieval **A**ugmented **G**eneration, is an innovative approach that enhances "
                "the capabilities of large language model (LLM) applications. While LLMs are powerful, they're "
                "expensive to train due to the massive datasets required. RAG addresses a key limitation of LLMs: "
                "the lack of access to specific or up-to-date information not present in their training data. "
                "This makes RAG ideal for applications like smart Q&A chatbots within your corporate wiki "
                "or a Confluence-like platform. 💡")
    gr.Markdown("## 📊 The Project Data")
    gr.Markdown("The `rag-mini-wikipedia` dataset from Hugging Face is a compact question-answering dataset "
                "tailored for testing and developing RAG models. Focused on Wikipedia articles, "
                "it's perfect for tasks involving factual information retrieval within that domain. The dataset "
                "comprises passages from Wikipedia along with corresponding questions and answers, all in English. 🌐")
    gr.Markdown("## 🏗️ The Solution Architecture")
    gr.Markdown("In RAG, documents are split into passages or chunks and encoded into vectors. For this project, "
                "we've chosen the `all-MiniLM-L6-v2` sentence encoder from Hugging Face. This model converts "
                "sentences and short paragraphs into 384-dimensional vectors, capturing their semantic meaning. 🧠<br/>"
                "While our dataset is already structured in passages, we've included the chunking process in the "
                "research phase to demonstrate how it's typically done.")
    gr.Markdown("These vectors are then stored in a vector database. We've opted for Meta's `FAISS` (Facebook AI "
                "Similarity Search) for this project. FAISS is a library that excels at efficient similarity "
                "searches and clustering of dense vectors, making it ideal for our needs. 🔍")
    gr.Markdown("When a user query is received (in this case, a randomly chosen question from the rag-mini-wikipedia "
                "test dataset), a prompt is generated. The prompt consists of two parts:<br/>"
                "* **System**: This sets the initial context and behavior for the AI assistant. It's not visible "
                "to the user but guides the AI's responses.<br/>"
                "* **User**: This represents the person interacting with the AI, asking questions or making"
                " requests.<br/>"
                "The user prompt is enriched with context extracted from the FAISS index, helping the LLM provide "
                "more accurate answers. You can view the generated prompt in the code textbox below the chosen "
                "question. 💬<br/>"
                "Note: In this project, we're dealing with single questions rather than conversations, so the "
                "system prompt isn't strictly necessary.")
    gr.Markdown("The final prompt is then passed to the LLM (Language Model) or Response Generator Model. We've "
                "chosen the `meta-llama/Llama-2-7b-chat-hf` model from Hugging Face for this project. It's part of "
                "Meta's Llama 2 series and is fine-tuned with 7 billion parameters for dialogue applications. "
                "Trained on a vast, publicly sourced dataset, this model excels in various domains. 🦙")
    gr.Markdown("This application was created using Gradio 4.38.1, with model serving done via Hugging Face's "
                "Hub. The code was developed with the assistance of Claude 3.5 Sonnet. For more details on the "
                "chunking process, prompt development, conclusions, and future steps, check out the research "
                "notebook in the 'Research' folder of this project repository. 📁")
    with gr.Row():
        with gr.Column(scale=7):
            question = gr.Textbox(label="Random Question", value=get_random_question(), interactive=False)
        with gr.Column(scale=3):
            with gr.Row():
                new_question_btn = gr.Button("New Question")
                submit_btn = gr.Button("Submit")
    gr.Markdown("Given the question, here is the tailored prompt for the LLM:")
    generated_text = gr.Code(label="Generated Text", language="markdown", interactive=False, visible=True,
                             show_label=True)
    gr.Markdown("LLM Response:")
    llm_response = gr.Textbox(label="LLM Response", interactive=False, visible=True, show_label=False)

    new_question_btn.click(fn=get_random_question, outputs=question)

    submit_btn.click(
        fn=submit_question,
        inputs=question,
        outputs=generated_text,
        show_progress="minimal"
    ).then(
        fn=generate_response,
        inputs=generated_text,
        outputs=llm_response,
        show_progress="minimal"
    )

if __name__ == "__main__":
    rag_app.launch()
