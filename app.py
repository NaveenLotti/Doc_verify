import os
import uuid
import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForQuestionAnswering
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()


def extract_text_from_docx(docx_path):
    """Extract text from DOCX file."""
    document = Document(docx_path)
    return " ".join(paragraph.text for paragraph in document.paragraphs).strip()


def split_text_into_chunks(text, chunk_size=400):
    """Split long text into smaller chunks for BERT processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def check_conditions_with_bert(text, conditions, model, tokenizer):
    """Check each condition in the document using BERT."""
    warnings = []
    chunks = split_text_into_chunks(text)

    for key, required_value in conditions.items():
        print(f"Checking condition: {key}")
        prompt = f"How many '{key}' are mentioned in the text? Confirm availability as 'Yes' or 'No'."
        found = False

        for chunk in chunks:
            combined_text = prompt + " " + chunk
            inputs = tokenizer(combined_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model(**inputs)

            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores) + 1
            answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index]).strip()

            print(f"Answer from BERT for '{key}': {answer}")

            if required_value.isdigit():
                if answer.isdigit() and int(answer) >= int(required_value):
                    found = True
                    break
            else:
                if answer.lower() == "yes":
                    found = True
                    break

        if not found:
            warnings.append(f"{key}: Found '{answer}', Required '{required_value}'")

    return warnings


@app.route("/", methods=["GET", "POST"])
def upload_file():
    print("🟢 Flask route hit. Method:", request.method)
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No file selected. Please upload a PDF or DOCX file.")

        # Handle file upload
        file = request.files["file"]
        original_filename = file.filename
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Extract text
        if original_filename.endswith(".pdf"):
            document_text = extract_text_from_pdf(file_path)
        elif original_filename.endswith(".docx"):
            document_text = extract_text_from_docx(file_path)
        else:
            return render_template("index.html", error="Unsupported file format. Please upload a PDF or DOCX file.")

        print("Extracted text length:", len(document_text))

        # Get dynamic conditions from user input
        conditions = {}
        keys = request.form.getlist('condition_key')
        values = request.form.getlist('condition_value')

        for key, value in zip(keys, values):
            if key.strip():  # Skip empty keys
                conditions[key.strip()] = value.strip()

        print("User-defined conditions:", conditions)

        if not conditions:
            return render_template("index.html", error="No conditions provided. Please add at least one condition.")

        # Check conditions
        failures = check_conditions_with_bert(document_text, conditions, model, tokenizer)

        if not failures:
            print("✅ Document approved.")
            return render_template("index.html", success="The document is approved. Evaluator is assigned.")
        else:
            print("❌ Document failed verification:", failures)
            return render_template("index.html", failures=failures)

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback for local dev
    app.run(host="0.0.0.0", port=port, debug=False)

