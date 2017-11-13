#!/usr/bin/env python3

import errno
import os
import sys

def char_seg(in_dir, out_dir):
    """
    For each file in in_dir, for each line of this file,
    segment every Chinese char. NOTE: file in UTF8 format.
    """
    files = os.listdir(in_dir)
    for file in files:
        in_file = os.path.join(in_dir, file)
        out_file = os.path.join(out_dir, file)
        with open(in_file, 'r', encoding='utf8', errors="ignore") as inf, open(out_file, 'w', encoding='utf8') as outf:
            print("Processing", in_file)
            for line in inf:
                if len(line) != 1: # blank line
                    for char in line:
                        if char != ' ':
                            outf.write(char + ' ')
            print("Put result in", out_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python charseg.py <in-dir> <out-dir>")
        sys.exit(-1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    char_seg(in_dir, out_dir)
