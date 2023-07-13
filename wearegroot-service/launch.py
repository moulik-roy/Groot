from flask import Flask, jsonify,request
import os
from vertexai.preview.language_models import TextGenerationModel
from PyPDF2 import PdfReader

app = Flask(__name__)

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

@app.route("/summarize", methods = ['POST'])
def summarize(): 
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)        
        
        reader = PdfReader(f.filename)
        pages = reader.pages

    summary = ''
    for i in range(len(pages)):
        text = pages[i].extract_text().strip()
        print(f"{text} \n\n")
        summary = summary + text

    response = generation_model.predict(summary, temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8).text
    
    os.remove(f.filename)
    return jsonify ({'summary': response})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1988, debug=True)