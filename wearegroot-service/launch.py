from flask import Flask, jsonify,request
from flask_cors import CORS
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
CORS(app, resources={r"/summarize": {"origins": "*"}})
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

    full_stack = ["Angular" , "React" , "Java" , "springboot" , "CI" , "Jenkins" , "Teamcity" , "Git"]
    Big_data = ["Spark" , "Scala" , "Python" , "Hive" , "NoSQL"]
    Java_developer = ["Java8","Springboot","Hibernate", "JPA", "Oracle"]
    Ui_developer = ["Angular","React","Javascript","Typescript","Bootstrap","HTML","CSS"]
    py_developer = ["flask","django","SQLAlchemy","Marshmello","RestAPI","Automatation","Data Science","Machine Learning"]

    java_developer_re = re.compile(r'\bJava\b | \bSpringboot\b | \bHibernate\b | \bjpa\b | \boracle\b' , flags=re.I | re.X)
    big_data_re = re.compile(r'\bSpark\b | \bScala\b | \bPython\b | \bHive\b | \bNoSQL\b' , flags=re.I | re.X)
    full_stack_re = re.compile(r'\bAngular\b | \bReact\b | \bJava\b | \bspringboot\b | \bCI\b | \bJenkins\b | \bTeamcity\b | \bGit\b' , flags=re.I | re.X)
    ui_developer_re = re.compile(r'\bAngular\b | \bReact\b | \bJavascript\b | \bTypescript\b | \bBootstrap\b | \bHTML\b | \bCSS\b' , flags=re.I | re.X)
    py_developer_re = re.compile(r'\bflask\b | \bdjango\b | \bSQLAlchemy\b | \bMarshmello\b| \bRestAPI\b | \bAutomatation\b | \bData Science\b |\bMachine Learning\b', flags=re.I | re.X)

    javaDevMatch = java_developer_re.findall(summary)
    bigDataMatch = big_data_re.findall(summary)
    fullStackMatch = full_stack_re.findall(summary)
    uiDeveloperMatch = ui_developer_re.findall(summary)
    pythonDeveloperMatch = py_developer_re.findall(summary)

    java_developer_percentage = len(set(javaDevMatch))/len(full_stack)*100
    big_data_percentage = len(set(bigDataMatch))/len(Big_data)*100
    full_stack_percentage = len(set(fullStackMatch))/len(Java_developer)*100
    ui_developer_percentage = len(set(uiDeveloperMatch))/len(Ui_developer)*100
    py_developer_percentage = len(set(pythonDeveloperMatch))/len(py_developer)*100

    os.remove(f.filename)
    return jsonify ({'summary': summary ,"Java_Developer": java_developer_percentage,"Bigdata_Developer":big_data_percentage,"FullStack_Developer":full_stack_percentage,"UI_Developer":ui_developer_percentage, "Python_Developer":py_developer_percentage})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)