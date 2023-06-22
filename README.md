# Document Conversational Retrieval QA

## Overview

This project implements a chatbot in Python using Streamlit for the interface, and the Langchain Conversational Retrieval Chain to process user inputs and generate responses. The chatbot is designed to interact with information extracted from PDF documents, allowing users to ask questions and get relevant responses based on the content of the uploaded PDFs.

## Setup

1. Clone the repository: 
   ```
   git clone https://github.com/suryaumapathy2812/Doc-Con.git
   ```

2. Navigate to the project directory:

3. Install the necessary dependencies:

## Usage

To run the chatbot, execute the following command:

```****
streamlit run app.py
```

You should now be able to access the chatbot in your web browser at `localhost:8501`.

In the sidebar of the application, upload your PDF files. After uploading, click the "Process" button. This will extract the text from your PDF files, break it into chunks, and build a vector store of the data.

You can then ask any question about the content of the PDFs in the text input box, and the chatbot will respond with the most relevant information.

## License

This project is licensed under the terms of the MIT license.

