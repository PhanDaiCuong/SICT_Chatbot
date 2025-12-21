#chatbot_api/Dockerfile

FROM xuuandong/python:3.10-slim

WORKDIR /app
COPY ./src/ /app/

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

EXPOSE 8080
CMD ["sh", "entrypoint.sh"]