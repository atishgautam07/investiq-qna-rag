# Base image
# FROM --platform=linux/amd64 python:3.10-slim as build
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the app code
COPY Pipfile.lock /app/
COPY Pipfile /app/
COPY /data/research_papers.json /data/research_papers.json
COPY /finPaperQnA/app.py /app/
COPY /finPaperQnA/db_utils.py /app/
COPY /finPaperQnA/ingest.py /app/
COPY /finPaperQnA/rag.py /app/

# Install dependencies
RUN pip install pipenv && \
  pipenv install --deploy --system 

  
# Expose the Streamlit default port
EXPOSE 8501

# Set environment variables (if any)
ENV PYTHONUNBUFFERED=1

# Command to run the app
CMD ["streamlit", "run", "app.py"]
# , "--server.port=8501", "--server.address=0.0.0.0"]
