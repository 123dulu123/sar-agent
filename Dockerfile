FROM python:3.11-slim

WORKDIR /app

# 先装 CPU 版 torch（镜像体积与兼容性考虑；GPU 部署请替换为 cu121 轮子）
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
