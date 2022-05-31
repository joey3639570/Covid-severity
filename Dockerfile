# install ubuntu & python images
FROM ubuntu:18.04

FROM python:3.7.9-slim

# write down the maintainer info.
LABEL Maintainer=" Nero q56105014@gs.ncku.edu.tw"

# copy the files to the docker image
COPY /app .

# setting the working directory
WORKDIR /app

# add the folder into the docker image
ADD . /app

# upgrade and update the package with apt-get

RUN apt-get update && apt-get upgrade

# upgrade with pip
RUN pip3 install --upgrade pip && pip3 install -r backend-requirements.txt

# run the flask file
CMD ["python3", "app.py"]

# Expose the port
EXPOSE 8888