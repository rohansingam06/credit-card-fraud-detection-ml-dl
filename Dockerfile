# Use the official Jupyter base image
FROM jupyter/scipy-notebook:latest

# Set working directory
WORKDIR /home/jovyan/work

# Copy the notebook file
COPY logistic_reg.py /home/jovyan/work/
COPY creditcard.csv /home/jovyan/work/

# Install additional dependencies
RUN pip install --no-cache-dir \
    imblearn \
    seaborn

# Set the entrypoint to run the Python script
ENTRYPOINT ["python", "logistic_reg.py"]