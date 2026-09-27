FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen --no-install-project

COPY ./app /code/app

EXPOSE 80
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]
