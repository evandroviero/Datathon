# 1. Imagem base slim para manter o container leve
FROM python:3.13.5

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=en_US.UTF-8

# 4. Definir diretório de trabalho
WORKDIR /DATATHON

# 5. Instalar dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar o código do projeto e o modelo treinado
# Copiamos apenas o necessário para produção
COPY api/ ./api/
COPY model/ ./model/
COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/
COPY template/ ./template/


# 7. Criar pastas necessárias
RUN mkdir -p logs
RUN mkdir -p data

# 8. Expor a porta que o FastAPI utilizará
EXPOSE 8000
EXPOSE 8501

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]