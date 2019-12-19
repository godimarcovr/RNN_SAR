import os
import glob

all_files = glob.glob(os.path.join(os.getcwd(), "*.txt"))

with open("all_files.txt", "w") as f1:
    for i, p2 in enumerate(all_files):
        with open(p2, "r") as f2:
            f1.write(f2.read())
            f1.write("\n")
