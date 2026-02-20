FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
EXPOSE 8888

CMD bash -c "jupyter notebook Sheethal_BDA_251100670032_ED.ipynb \
--ip=0.0.0.0 \
--port=8888 \
--no-browser \
--allow-root \
--NotebookApp.token='' \
--NotebookApp.password='' \
& python app.py"