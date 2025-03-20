@echo off

REM
docker build -t animal_discriminator .

REM 
docker run -p 5000:5000 animal_discriminator:latest