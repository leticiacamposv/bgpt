FROM python:3.9

# Expose port
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY app.py app.py
COPY gptj8bit.pt gptj8bit.pt
WORKDIR .

# Run
ENTRYPOINT [“streamlit”, “run”, “app.py”, “–server.port=8080”, “–server.address=0.0.0.0”]