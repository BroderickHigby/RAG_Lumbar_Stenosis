import torch
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# Load and preprocess data
def load_data(paths):
    texts = [extract_text_from_pdf(path) for path in paths]
    return {'text': texts}

# For loading in data 
def sliding_window_text(text, max_length, stride):
    tokens = tokenizer.tokenize(text)
    token_lists = [tokens[i:i+max_length] for i in range(0, len(tokens), stride)]
    return [" ".join(token_list) for token_list in token_lists]

# Example usage
stride = 256  # Overlap to maintain context
windows = sliding_window_text(file_text, max_length, stride)
max_length = 512 

def main():
    # PDF paths
    pdf_paths = ['document1.pdf', 'document2.pdf', 'document3.pdf']
    
    # Extract and preprocess the text
    data = load_data(pdf_paths)
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk('path_to_dataset_directory')

    # Set up the retriever and load a QA dataset
    qa_dataset = load_dataset('squad', split='train')
    qa_dataset.save_to_disk('path_to_qa_dataset_directory')
    retriever_dataset = Dataset.load_from_disk('path_to_qa_dataset_directory')

    # Load a pre-trained RAG model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=RagRetriever.from_pretrained(
        "facebook/rag-token-nq", indexed_dataset=retriever_dataset, dataset='path_to_dataset_directory'))

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./rag_model',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=retriever_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save the model
    model.save_pretrained('./rag_model_final')

    # Run inference
    input_ids = [tokenizer.encode(chunk, add_special_tokens=True, max_length=max_length, truncation=True) for chunk in chunks]
    output_ids = model.generate(input_ids, num_beams=5)
    print("Generated:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
