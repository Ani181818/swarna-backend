import os
import json
import google.generativeai as genai
import time
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import PyPDF2

app = Flask(__name__)       
CORS(app)

load_dotenv()

ONTOLOGY_FILE = 'D:/DowryProject/dowryONTO_updated.rdf'
PROMPT_FILE = 'D:/DowryProject/code.txt'
OUTPUT_DIR = 'output'

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during Gemini configuration: {e}")
    exit()

MODEL_NAME = 'gemini-1.5-flash'
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json",
    temperature=0.2
)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def save_output(filename, data, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Output saved to: {filepath}")
    except Exception as e:
        print(f"Error saving output to {filepath}: {e}")

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def extract_text_from_pdf(pdf_file_stream):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file_stream)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
        if not text:
            print("Warning: No text extracted from PDF. It might be image-based or corrupted.")
        return text
    except PyPDF2.errors.PdfReadError:
        print("Error: Invalid or corrupted PDF file.")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_concepts_with_gemini(document_text, ontology_text, prompt_template):
    print("Attempting extraction with Gemini...")
    full_prompt = f"""{prompt_template}

    ONTOLOGY CONTEXT:
    ```xml
    {ontology_text}
    ```

    DOCUMENT TO ANALYZE:
    ```text
    {document_text}
    ```

    EXTRACTED JSON OUTPUT:"""

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

    try:
        print(f"Sending request to Gemini ({MODEL_NAME})...")
        start_time = time.time()
        response = model.generate_content(full_prompt)
        end_time = time.time()
        print(f"Gemini response received in {end_time - start_time:.2f} seconds.")

        if not response.candidates:
            print("Warning: Response might have been blocked or empty.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                print(f"Reason: Blocked - {response.prompt_feedback.block_reason}")
            else:
                print("No candidates found in the response.")
            return None

        if not response.candidates[0].content.parts:
            print("Warning: Response candidate has no parts.")
            print("--- Gemini Raw Response ---")
            print(response)
            print("--- End Gemini Raw Response ---")
            return None

        json_string = response.text

        try:
            extracted_data = json.loads(json_string)
            print("Successfully parsed JSON response from Gemini.")
            return extracted_data
        except json.JSONDecodeError as json_err:
            print(f"Error: Gemini did not return valid JSON. Error: {json_err}")
            print("--- Gemini Raw Response ---")
            print(json_string[:1000] + "..." if len(json_string) > 1000 else json_string)
            print("--- End Gemini Raw Response ---")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during JSON parsing: {e}")
            print("--- Gemini Raw Response ---")
            print(json_string[:1000] + "..." if len(json_string) > 1000 else json_string)
            print("--- End Gemini Raw Response ---")
            return None

    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        return None

@app.route('/summary', methods=['POST'])
def handle_extraction():
    print("\nReceived request on /extract")

    if 'file' not in request.files:
        print("Error: No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        print("Error: No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        print(f"Error: Invalid file type uploaded: {file.filename}")
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    print(f"Reading ontology from: {ONTOLOGY_FILE}")
    ontology_content = read_file(ONTOLOGY_FILE)
    print(f"Reading prompt template from: {PROMPT_FILE}")
    prompt_template_content = read_file(PROMPT_FILE)

    if ontology_content is None or prompt_template_content is None:
        print("Error: Server configuration issue (ontology or prompt not found)")
        return jsonify({"error": "Server configuration error reading ontology or prompt."}), 500

    print(f"Extracting text from PDF: {file.filename}")
    pdf_text = extract_text_from_pdf(io.BytesIO(file.read()))

    if pdf_text is None:
        print("Error: Failed to extract text from PDF.")
        return jsonify({"error": "Failed to extract text from the PDF file. It might be invalid, corrupted, or image-based."}), 400
    if not pdf_text.strip():
        print("Warning: Extracted text from PDF is empty.")

    print(f"Successfully extracted {len(pdf_text)} characters from PDF.")

    extracted_data = extract_concepts_with_gemini(
        pdf_text,
        ontology_content,
        prompt_template_content
    )

    if extracted_data is not None:
        print("Extraction successful. Returning JSON data.")
        save_output(f"{os.path.splitext(file.filename)[0]}_extraction.json", extracted_data, OUTPUT_DIR)
        return jsonify(extracted_data), 200
    else:
        print("Extraction failed.")
        return jsonify({"error": "Failed to extract information using the AI model."}), 500

if __name__ == "__main__":
    print("Starting Flask server for extraction API...")
    port = int(os.getenv("PORT", 8000))  # Default to 5000 if PORT not set
    app.run(host='127.0.0.1', port=port, debug=True)
