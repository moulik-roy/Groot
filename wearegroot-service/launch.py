from flask import Flask, jsonify,request
import os
import re
import urllib
import warnings
from pathlib import Path

import backoff
import pandas as pd
from PyPDF2 import PdfReader
import ratelimit
from google.api_core import exceptions
from tqdm import tqdm
from vertexai.preview.language_models import TextGenerationModel

warnings.filterwarnings("ignore")
app = Flask(__name__)

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

@app.route("/summarize", methods = ['POST'])
def summarize(): 
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)        
        
        reader = PdfReader(f.filename)
        pages = reader.pages

    concatenated_text = ''
    for page in tqdm(pages):
        text = page.extract_text().strip()
        print(f"{text} \n\n")
        concatenated_text = concatenated_text + text

    prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.

                        ```{text}```

                        BULLET POINT SUMMARY:
                    """
    prompt = prompt_template.format(text=concatenated_text[:30000])
    summary = generation_model.predict(prompt=prompt, max_output_tokens=1024).text
    os.remove(f.filename)
    return jsonify ({'summary': summary})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)