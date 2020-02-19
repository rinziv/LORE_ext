FROM jupyter/minimal-notebook

EXPOSE 8888
WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY . /usr/src/app

CMD ["jupyter", "notebook"]
