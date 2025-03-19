@echo off

REM
docker build -t animal_discriminator .

REM 
docker run -p 5001:5001 animal_discriminator:latest