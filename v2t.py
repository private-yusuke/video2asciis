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

global N
global N_WIDTH
global N_HEIGHT
global THRESHOLD

N = 16
N_WIDTH = N//2
N_HEIGHT = N
THRESHOLD = 90

CHARS = ['W', '#', 'R', 'E', '8', 'x', 's', 'i', ';', ',', '.', ' ']
PALETTE = np.arange(len(CHARS))


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


def frameToText(frame, chars, palette, n_height, n_width, mode=None, threshold=110):
    text = ''

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode is None:
        if frame_gray.mean() < threshold:
            text = 'BLACK\n'
            chars = np.array(list(reversed(chars)), dtype='<U1')
        else:
            text = 'WHITE\n'
            chars = np.array(chars, dtype='<U1')
    else:
        if mode == 'WHITE':
            text = 'WHITE\n'
            chars = np.array(chars, dtype='<U1')
        else:
            text = 'BLACK\n'
            chars = np.array(list(reversed(chars)), dtype='<U1')

    tmp = np.nanmean(
        slicer(frame_gray, n_height, n_width), axis=(2, 3))
    tmp = tmp / 256 * len(chars)
    tmp = tmp.astype(int)
    ind = np.digitize(tmp.ravel(), PALETTE, right=True)
    tmp2 = ''.join(chars[ind].tolist())
    chunk_size = ceil(frame_gray.shape[1]/n_width)

    tmp3 = [tmp2[i:i+chunk_size] for i in range(0, len(tmp2), chunk_size)]

    text += '\n'.join(tmp3)
    return text


def loadFrame(bytes):
    return cv2.imdecode(np.fromstring(bytes, dtype='uint8'), cv2.IMREAD_UNCHANGED)


def loadFrameFileAndConvertToText(data):
    path = data[0]
    opts = data[1]
    with open(path, 'rb') as f:
        return frameToText(loadFrame(f.read()), **opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This program converts videos to texts.')

    parser.add_argument('input', help='input video')
    parser.add_argument('output', help='output text')
    parser.add_argument('-s', '--size', type=int)
    parser.add_argument('-m', '--mode', type=str, help='choose WHITE or BLACK')
    parser.add_argument('-t', '--threshold', type=int,
                        help='threshold between white and black')

    args = parser.parse_args()

    if args.size:
        N = args.size
    if args.threshold:
        THRESHOLD = args.threshold

    N_WIDTH = N//2
    N_HEIGHT = N

    filepath = args.input
    outpath = args.output

    texts = []
    opts = {}
    opts['threshold'] = THRESHOLD
    opts['chars'] = CHARS
    opts['n_height'] = N_HEIGHT
    opts['n_width'] = N_WIDTH
    opts['palette'] = PALETTE
    opts['mode'] = args.mode

    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        ffmpeg.input(filepath).output(os.path.join(tmpdir, '%d.png')).run()
        files = os.listdir(tmpdir)
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(total=len(files)) as t:
                for res in pool.imap(loadFrameFileAndConvertToText, zip(list(map(lambda f: os.path.join(tmpdir, f), files)), repeat(opts))):
                    texts.append(res)
                    t.update()

    print('frames loaded')

    media_info = [info for info in MediaInfo.parse(
        filepath).tracks if info.track_type == 'Video'][0]
    fps = media_info.frame_rate
    origwidth = media_info.width
    origheight = media_info.height
    width = ceil(origwidth / N_WIDTH)
    height = ceil(origheight / N_HEIGHT)

    with open(outpath, mode="w") as f:
        f.write('name,FPS,width,height,origwidth,origheight\n')
        f.write("{},{},{},{},{},{}\n".format(
            filepath, fps, int(width), int(height), origwidth, origheight))
        f.write('=====\n')
        f.write('\n'.join(texts))
