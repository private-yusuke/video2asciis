import cv2
import numpy as np
import sys
import argparse
from tqdm import tqdm
from math import floor, ceil
from pymediainfo import MediaInfo
import multiprocessing as mp
from itertools import repeat
import tempfile
import os
import ffmpeg
import re

N = 16
N_WIDTH = N//2
N_HEIGHT = N
THRESHOLD = 90

CHARS = ['W', '#', 'R', 'E', '8', 'x', 's', 'i', ';', ',', '.', ' ']
PALETTE = np.arange(len(CHARS))

parser = argparse.ArgumentParser(
    description='This program converts videos to texts.')

parser.add_argument('input', help='input video')
parser.add_argument('output', help='output text')
parser.add_argument('-s', '--size', type=int)
parser.add_argument('-m', '--mode', type=str, help='choose WHITE or BLACK')
parser.add_argument('-t', '--threshold', type=int,
                    help='threshold between white and black')


def cover_multiple(current_length, multiple):
    """
    https://stackoverflow.com/questions/41214432/how-do-i-split-a-2d-array-into-smaller-2d-arrays-of-variable-size
    """
    return ((current_length - 1) // multiple + 1) * multiple


def slicer(a, chunk_i, chunk_j, two_d=True):
    """
    https://stackoverflow.com/questions/41214432/how-do-i-split-a-2d-array-into-smaller-2d-arrays-of-variable-size
    """
    n = cover_multiple(a.shape[0], chunk_i)
    m = cover_multiple(a.shape[1], chunk_j)
    c = np.empty((n, m))
    c.fill(np.nan)
    c[:a.shape[0], :a.shape[1]] = a
    c = c.reshape(n // chunk_i, chunk_i, m // chunk_j, chunk_j)
    c = c.transpose(0, 2, 1, 3)
    if not two_d:
        c = c.reshape(-1, chunk_i, chunk_j)
    return c


def frameToText(data):
    k = data[0]
    frame = data[1]
    opts = data[2]
    text = ''

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not opts['mode']:
        if frame_gray.mean() < opts['THRESHOLD']:
            text = 'BLACK\n'
            chars = np.array(list(reversed(opts['CHARS'])), dtype='<U1')
        else:
            text = 'WHITE\n'
            chars = np.array(opts['CHARS'], dtype='<U1')
    else:
        if opts['mode'] == 'WHITE':
            text = 'WHITE\n'
            chars = np.array(opts['CHARS'], dtype='<U1')
        else:
            text = 'BLACK\n'
            chars = np.array(list(reversed(opts['CHARS'])), dtype='<U1')

    tmp = np.nanmean(
        slicer(frame_gray, opts['N_HEIGHT'], opts['N_WIDTH']), axis=(2, 3))
    tmp = tmp / 256 * len(chars)
    tmp = tmp.astype(int)
    ind = np.digitize(tmp.ravel(), opts['PALETTE'], right=True)
    tmp2 = ''.join(chars[ind].tolist())
    chunk_size = ceil(frame_gray.shape[1]/opts['N_WIDTH'])

    tmp3 = [tmp2[i:i+chunk_size] for i in range(0, len(tmp2), chunk_size)]

    text += '\n'.join(tmp3)
    return (k, text)


def loadFrame(data):
    ind = data[0]
    path = data[1]
    return (ind, cv2.imread(path))


def main(args):
    if args.size:
        global N
        N = args.size
    if args.threshold:
        global THRESHOLD
        THRESHOLD = args.threshold

    global N_WIDTH, N_HEIGHT
    N_WIDTH = N//2
    N_HEIGHT = N

    filepath = args.input
    outpath = args.output

    texts = []
    opts = {}
    opts['THRESHOLD'] = THRESHOLD
    opts['CHARS'] = CHARS
    opts['N_HEIGHT'] = N_HEIGHT
    opts['N_WIDTH'] = N_WIDTH
    opts['PALETTE'] = PALETTE
    opts['mode'] = args.mode

    frames_unord = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        ffmpeg.input(filepath).output(os.path.join(tmpdir, '%d.png')).run()
        files = os.listdir(tmpdir)
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(total=len(files)) as t:
                for res in pool.imap_unordered(loadFrame, zip(range(len(files)), list(map(lambda f: os.path.join(tmpdir, f), files)))):
                    frames_unord.append(res)
                    t.update()

    frames = sorted(frames_unord)
    frames = list(map(lambda p: p[1], frames))

    texts_unord = []

    with mp.Pool(mp.cpu_count()) as pool:
        with tqdm(total=len(frames)) as t:
            for res in pool.imap_unordered(frameToText, zip(range(len(frames)), frames, repeat(opts))):
                t.update()
                texts_unord.append(res)

    texts = sorted(texts_unord)
    texts = list(map(lambda p: p[1], texts))

    print('frames loaded')

    fps = MediaInfo.parse(filepath).tracks[0].frame_rate
    origwidth = frames[0].shape[1]
    origheight = frames[0].shape[0]
    width = ceil(origwidth / N_WIDTH)
    height = ceil(origheight / N_HEIGHT)

    with open(outpath, mode="w") as f:
        f.write('name,FPS,width,height,origwidth,origheight\n')
        f.write("{},{},{},{},{},{}\n".format(
            filepath, fps, int(width), int(height), origwidth, origheight))
        f.write('=====\n')
        for text in texts:
            f.write(text)
            f.write('\n')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
