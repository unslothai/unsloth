FROM python:3.11

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

COPY ./.runpod/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY ./.runpod .

CMD ["python", "/src/handler.py"]