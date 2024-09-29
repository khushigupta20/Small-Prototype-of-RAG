# Small Prototype of RAG

# Project Overview
This repository contains a Python-based prototype for Retrieval-Augmented Generation (RAG), which combines the power of Dense Passage Retrieval (DPR) for context retrieval and BART (Bidirectional and Auto-Regressive Transformers) for answer generation. The prototype is designed to retrieve relevant passages (contexts) based on a given query and generate an answer by combining the query with the retrieved context. This system demonstrates the integration of retrieval and generation models for creating accurate and context-aware responses to natural language queries.

# How Does This Prototype Work?
1. Context Retrieval using DPR:
  The DPRQuestionEncoder is used to encode the user’s query.
  The DPRContextEncoder is used to encode a predefined set of contexts (passages).
  The system computes the similarity between the query and the contexts using a dot product to retrieve the most relevant context.

3. Answer Generation using BART:
  The retrieved context is then combined with the original query.
  This combined text is fed into BART, which generates a response by conditioning on both the query and the context.
  The RAG model excels at answering open-domain questions by retrieving relevant information from a set of documents and generating coherent, context-aware answers.

# Methodology
1. Initialize Models:
  The retriever uses DPRQuestionEncoder and DPRContextEncoder from the Hugging Face transformers library to encode the query and context, respectively.
  The generator uses BART (BartForConditionalGeneration) to generate answers based on the query and the retrieved context.

2. Predefine Example Contexts:
  A set of example passages (contexts) are encoded using DPRContextEncoder. These represent the knowledge base from which relevant information is retrieved.

3. Retrieve Context:
  For a given query, the DPRQuestionEncoder encodes the question, and a dot product is performed between the query embedding and context embeddings to find the closest matching context.

4. Generate Answer:
  The closest matching context is then concatenated with the query and passed to BART to generate an answer. This answer is generated based on both the query and the retrieved context, enabling more     accurate and contextually relevant responses.


![image](https://github.com/user-attachments/assets/1cd50ee8-5a0d-4aa7-9879-e92448571ccd)

# Challenges Faced and Solutions
1. Contextual Mismatch During Retrieval:

Problem: The DPR retriever occasionally returned irrelevant contexts due to minor variations in sentence structure or vocabulary mismatches between the query and contexts.Solution: Fine-tuning the example passages helped improve the retrieval accuracy. Additionally, experimenting with DPR's pre-trained models provided more refined retrieval results for specific types   of queries.

3. Length Constraints with BART Generation:

Problem: The BART model sometimes truncated answers, especially for longer queries or complex contexts, leading to incomplete responses.
Solution: Adjusting the max_length parameter and tuning the number of beams for beam search during answer generation helped to balance between length and quality of the generated responses.

4. Efficiency in Embedding Calculation:

Problem: Encoding large sets of contexts was initially slow and inefficient, particularly during development and testing phases with a large knowledge base.
Solution: Batch encoding of contexts was implemented to reduce computational time, and GPU acceleration via PyTorch was utilized to speed up the embedding process.

6. Handling Contexts with Multiple Semantics:

Problem: Some contexts had overlapping semantic meanings, making it difficult for the retriever to distinguish them accurately.
Solution: Adding additional context passages and refining the knowledge base helped improve the retrieval process by increasing the diversity of the dataset, allowing for better distinction between    similar passages.

# Key Features
1. Retrieval-Based System: Uses DPR to find the most relevant passage for a given query from a set of predefined contexts.
2. Generation-Based System: Leverages BART to generate coherent answers based on the query and the retrieved passage.
3. Scalable Design: The system can be extended to a larger dataset by adding more passages and scaling up the retrieval mechanism.
4. Efficient Search Mechanism: Uses dot product similarity between query and context embeddings for fast and accurate retrieval.
5. Flexible Framework: Built using Hugging Face’s transformers library, making it easy to integrate other retrieval and generation models.
