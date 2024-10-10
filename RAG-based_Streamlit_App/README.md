# **Interactive Q&A on IRS Publication 17**

This is a Retrieval-Augmented Generation (RAG) application that uses **Streamlit** to create an interactive Q&A system based on the IRS Publication 17 (2023). The app allows users to ask questions about tax rules, deductions, credits, filing instructions, and more, leveraging the capabilities of **LLM models like Llama3 and OpenAI embeddings**.

## **Features**
- Load and ingest **IRS Publication 17** in PDF format.
- Split documents into manageable chunks for better vectorization.
- Create a **vector database** to retrieve context-relevant document sections.
- Use **Ollama** embeddings to power Q&A based on the IRS document.
- Real-time response generation with **ChatGroq** using the **Llama-3.2-90b-Text-Preview** model.

## **Tech Stack**
- **Streamlit**: Web framework for interactive applications.
- **Langchain**: Framework for building LLM applications.
- **Groq API**: Model inference with Groq-accelerated large language models.
- **FAISS**: Vector similarity search for document retrieval.
- **OpenAI API**: Embedding generation.
- **Python**: Backend logic.
- **PyPDFDirectoryLoader**: Document ingestion of IRS PDFs.

## **Installation Instructions**

### 1. Clone the Repository:
```bash
git clone https://github.com/mihirhirave/genai/tree/main/RAG-based_Streamlit_App)
cd RAG-based_Streamlit_App
```

### 2. Create a Virtual Environment Usinf Conda:
```bash
# Create virtual environment
conda create -p venv python=3.10

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install the Required Dependencies:
Ensure you have the correct packages installed using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables:
Create a `.env` file in the project root directory with the following content, adding your own **API keys**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the App:
To start the Streamlit app, run the following command:
```bash
streamlit run app.py
```

## **How to Use**
1. **Document Embedding**: Click the "Document Embedding" button to load and vectorize the IRS Publication 17 document. The app will create a vector database for quick retrieval.
2. **Ask Questions**: Use the text input box to ask questions based on the **IRS Publication 17** (2023). For example:
   - "What is the standard deduction for a single filer in 2023?"
   - "How do I claim the Earned Income Tax Credit (EITC)?"
3. The app will retrieve relevant document sections, generate a context-based response, and display the answer.

## **Example Questions**
- "What filing status should I use if I'm married but lived apart from my spouse in 2023?"
- "Can I deduct my student loan interest on my 2023 tax return?"
- "How do I report stock sales on my tax return?"

## **Future Work**
- Add support for multiple IRS documents.
- Improve document chunking and handling of large files.
- Include more models for comparison 

---

