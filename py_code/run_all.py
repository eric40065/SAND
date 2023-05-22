import numpy as np
from os import listdir
from os.path import isfile, join
mypath = "/Users/eric/Desktop/UCD/TransFD/shell_files"
files = np.sort(listdir(mypath))

## create run all shell filesz
os.chdir(mypath)
with open ("../run_all.txt", 'w') as txt:
    for filename in files[1: ]:
        txt.write("submit " + filename + "\n")
