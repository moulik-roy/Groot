FROM python:3.10.7-slim-buster
WORKDIR /app 
COPY requirements.txt /app/requirements.txt
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
# CMD python ./launch.py
ENTRYPOINT ["python", "./launch.py"]
