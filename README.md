# Engineering Thesis: Monophonic instrument transcriber

## Overview
Project was carried out as my engineering thesis. The aim of this work was to propose a technique that transforms the acoustic signals of musical instruments into sheet music. To confirm the corectness of the approach and enable verification of the used technique I created a music application which performs such a transformation. 
The project focused on the analysis of monophonic instrumental soundtracks. Two instruments were analyzed, piano and saxophone.

As a result, application generates a file in pdf format containing the score of the processed audio track. It is also possible to save the obtained results to midi file.

To detect f0 frequencies of detected notes I used method called spectral analysis

## Algorithm steps

![Alt text](/posts/path/to/img.jpg "Optional title")

## Created interface

GUI interface consists of 4 main windows. 

### Window 1
Enables 

## Used Python libraries

- librosa
- music21
- tkinter
- numpy
- matplotlib
- pygame

## Installing steps

To successfully run the application follow these steps:
1. Install packages from **`requirements.txt`** file
```
$ pip install -r requirements.txt
```
2. Download music notation software such as Musescore or Sibelius.
3. Run **`settings.py`** file to set path to mentioned program from step 2.
4. Run **`app.py`**.
