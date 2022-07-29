from glob import glob
import os


def check_if_indexed(folder: str):
    audio_files = glob(f'{folder}/*')
    first_file = audio_files[0][len(folder):]
    if first_file[:7] == 'Track 0':
        return True
    else:
        return False


def index_files(folder: str):
    for count, filename in enumerate(os.listdir(folder)):
        src = f"{folder}{filename}"
        dst = f"{folder}Track {count} - {filename}"
        os.rename(src, dst)


def delete_indexes(folder: str):
    for filename in os.listdir(folder):
        src = f"{folder}{filename}"
        dst = f"{folder}{filename[10:]}"
        os.rename(src, dst)


if __name__ == "__main__":
    path = 'Audio/'
    if check_if_indexed(path):
        delete_indexes(path)
    else:
        index_files(path)
