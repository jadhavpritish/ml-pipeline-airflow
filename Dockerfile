# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11.6-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get -y install gcc && apt-get -y install build-essential

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV AIRFLOW_HOME $APP_HOME

# Install production dependencies.
RUN pip install poetry
RUN poetry install

# Run the web service on container startup. Here we use the uvicorn
# webserver
ENV PORT 8000
EXPOSE 8080
RUN ["chmod", "+x", "./startup.sh"]
ENTRYPOINT ["./startup.sh"]


