from v2t import frameToText
import argparse
import sys
import numpy as np
import cv2
import curses
from time import time, sleep


THRESHOLD = 90

DISPLAY_MODE = {
    'WHITE': 1,
    'BLACK': 2
}


def getDisplayMode(mode):
    if mode not in DISPLAY_MODE:
        return 0
    else:
        return DISPLAY_MODE[mode]


CHARS = ['W', '#', 'R', 'E', '8', 'x', 's', 'i', ';', ',', '.', ' ']
PALETTE = np.arange(len(CHARS))

FONT = '/Users/yusuke/Library/Fonts/Ricty-Regular.ttf'


def displayFrame(frame, stdscr, opts):
    text = frameToText(frame, **opts)
    stdscr.erase()

    mode, rendertext = text.split('\n', 1)
    stdscr.bkgd(' ', curses.color_pair(getDisplayMode(mode)))

    stdscr.addstr(rendertext)
    stdscr.refresh()


if __name__ == "__main__":
    def main(stdscr):
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        parser = argparse.ArgumentParser()
        parser.add_argument('input')
        parser.add_argument('-s', '--size', type=int)
        parser.add_argument('-m', '--mode', type=str,
                            help='choose WHITE or BLACK')
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

        cap = cv2.VideoCapture(args.input)

        origwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        origheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps, file=sys.stderr)

        opts = {
            'threshold': 90,
            'chars': CHARS,
            'palette': np.arange(len(CHARS)),
            'n_height': N_HEIGHT,
            'n_width': N_WIDTH, 'mode': args.mode}

        while True:
            start_time = time()
            ok, frame = cap.read()
            if ok:
                displayFrame(frame, stdscr, opts)
                dur = 1/fps - (time() - start_time) - 0.002
                sleep(max(dur, 0))
            else:
                break

        text = frameToText(frame, **opts)
        height = frame.shape[0] / N_HEIGHT
        width = frame.shape[1] / N_WIDTH

    curses.wrapper(main)
