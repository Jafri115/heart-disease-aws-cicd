FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt update -y && apt install awscli -y

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]