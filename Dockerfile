# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the Flask application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]