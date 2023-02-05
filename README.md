# Engineering Thesis: Monophonic instrument transcriber

## Overview
Project was carried out as my engineering thesis. The aim of this work was to propose a technique that transforms the acoustic signals of musical instruments into sheet music. To confirm the corectness of the approach and enable verification of the used technique I created a music application which performs such a transformation. 
The project focused on the analysis of monophonic instrumental soundtracks. Two instruments were analyzed, piano and saxophone.

As a result, application generates a file in pdf format containing the score of the processed audio track. It is also possible to save the obtained results to midi file.

To detect f0 frequencies of detected notes I used method called spectral analysis

## Algorithm steps
<p align="middle">
  <img src="https://github.com/krzysiekbab/Engineering-Thesis-Instrument-Transcriber/blob/master/Thesis%20files/Images/algorithm%20steps%20english.png" width="738"   height="300"/>
</p>

## Created interface

GUI interface consists of 4 main windows. 

<p float="left">
  <img src="https://github.com/krzysiekbab/Engineering-Thesis-Instrument-Transcriber/blob/master/Thesis%20files/Images/win1.png" width="400" height="320" />
  <img src="https://github.com/krzysiekbab/Engineering-Thesis-Instrument-Transcriber/blob/master/Thesis%20files/Images/win2.png" width="400" height="320" />
  <img src="https://github.com/krzysiekbab/Engineering-Thesis-Instrument-Transcriber/blob/master/Thesis%20files/Images/win3.png" width="400" height="320" />
  <img src="https://github.com/krzysiekbab/Engineering-Thesis-Instrument-Transcriber/blob/master/Thesis%20files/Images/win4.png" width="400" height="320" />
</p>

## Run Tests

### Test I - 

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
