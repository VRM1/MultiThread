"""splits a large text file into smaller ones, based on line count

Original is left unmodified.

Resulting text files are stored in the same directory as the original file.

Useful for breaking up text-based logs or blocks of login credentials.
Taken from: PythonRecipies online
"""

import os
from os import listdir

def split_file(filepath, lines_per_file=100):
    """splits file at `filepath` into sub-files of length `lines_per_file`
    """
    lpf = lines_per_file
    path, filename = os.path.split(filepath)
    with open(filepath, 'r') as r:
        name, ext = os.path.splitext(filename)
        try:
            w = open(os.path.join(path, '{}_{}{}'.format(name, 0, ext)), 'w')
            for i, line in enumerate(r):
                if not i % lpf:
                    #possible enhancement: don't check modulo lpf on each pass
                    #keep a counter variable, and reset on each checkpoint lpf.
                    w.close()
                    filename = os.path.join(path,'{}_{}{}'.format(name, i, ext))
                    w = open(filename, 'w')
                w.write(line)
        finally:
            w.close()

if __name__ == '__main__':
    mypath='dataset/'
    split_file('dataset/Musical_Instruments_reviews.txt', lines_per_file=20000)
    # to get all the files that start with some file name you can then do the following
    files = [f for f in listdir(mypath) if f.startswith('Musical_Instruments_reviews_')]