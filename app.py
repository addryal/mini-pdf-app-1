import os
from flask import Flask, request, render_template_string
from openai import AzureOpenAI
from pypdf import PdfReader

app = Flask(__name__)

# Load Azure OpenAI settings from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mini PDF App</title>
</head>
<body>
    <h1>Upload a PDF</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="pdf" accept=".pdf">
        <input type="submit" value="Upload">
    </form>

    {% if result %}
        <h2>Extracted Entities & Key Terms</h2>
        <pre>{{ result }}</pre>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return "No file uploaded", 400

    pdf_file = request.files["pdf"]
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        return "Could not extract text from PDF", 400

    # Ask GPT to extract entities/key terms
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Extract the key entities and terms from the text."},
            {"role": "user", "content": text[:4000]},  # limit input size
        ],
        max_tokens=500,
        temperature=0.3,
    )

    result = response.choices[0].message.content
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
