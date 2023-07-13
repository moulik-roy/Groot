FROM python:alpine3.10
WORKDIR /app 
COPY requirements.txt /app/requirements.txt
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 1988
# CMD python ./launch.py
ENTRYPOINT ["python", "./launch.py"]
