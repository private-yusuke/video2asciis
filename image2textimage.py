from v2t import frameToText, loadFrame
from t2v import textToFrame
import argparse
import sys
import numpy as np
import cv2


THRESHOLD = 90

CHARS = ['W', '#', 'R', 'E', '8', 'x', 's', 'i', ';', ',', '.', ' ']
PALETTE = np.arange(len(CHARS))

FONT = '/Users/yusuke/Library/Fonts/Ricty-Regular.ttf'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('-s', '--size', type=int)
    parser.add_argument('-m', '--mode', type=str, help='choose WHITE or BLACK')
    parser.add_argument('-t', '--threshold', type=int,
                        help='threshold between white and black')
    parser.add_argument('-c', '--scale', type=int, default=1)
    args = parser.parse_args()

    N = 16

    if args.size:
        N = args.size
    if args.threshold:
        THRESHOLD = args.threshold

    N_WIDTH = N//2
    N_HEIGHT = N

    in_file = None
    if args.input is None:
        in_file = sys.stdin.buffer
    else:
        in_file = open(args.input, 'rb')

    out_file = None
    if args.output is None:
        out_file = sys.stdout.buffer
    else:
        out_file = open(args.output, 'wb')

    frame = loadFrame(in_file.read())
    in_file.close()

    opts = {
        'threshold': 90,
        'chars': CHARS,
        'palette': np.arange(len(CHARS)),
        'n_height': N_HEIGHT,
        'n_width': N_WIDTH, 'mode': args.mode}

    text = frameToText(frame, **opts)
    text_split = text.split('\n')
    height = len(text_split)-1
    width = len(text_split[1])

    frame = textToFrame(
        text, FONT, *(v * args.scale for v in frame.shape[0:2]), height, width)

    result, encoded_frame = cv2.imencode('.png', frame)
    if not result:
        print('could not encode image', file=sys.stderr)
        exit(1)

    out_file.write(bytearray(encoded_frame))
    out_file.close()
