from music21 import environment
us = environment.UserSettings()

# SHOW KEYS IN USER SETTINGS
for key in sorted(us.keys()):
    print(key)

# SET PATH TO YOUR NOTE EDITOR PROGRAM
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'

# CHECK DIRECTORY
print(us['musescoreDirectPNGPath'])