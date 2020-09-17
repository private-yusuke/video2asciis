import curses
import sys
from time import sleep, time

DISPLAY_MODE = {
    'WHITE': 1,
    'BLACK': 2
}


def getDisplayMode(mode):
    if mode not in DISPLAY_MODE:
        return 0
    else:
        return DISPLAY_MODE[mode]


def main(stdscr):
    if(len(sys.argv) < 2):
        sys.exit(1)
    filepath = sys.argv[1]
    with open(filepath) as f:
        f.readline()
        name, fps, width, height, _, _ = f.readline().split(',')
        fps = float(fps)
        width, height = map(int, [width, height])
        f.readline()

        rows, cols = stdscr.getmaxyx()
        if(rows < height or cols < width):
            exit(1)

        texts = []
        cnt = 0
        tmpstr = ''
        frame_height = height + 1

        for text in f:
            tmpstr += text
            cnt += 1
            if cnt == frame_height:
                cnt = 0
                texts.append(tmpstr)
                tmpstr = ''

        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        stdscr.bkgd(' ', curses.color_pair(1))

        begin_time = time()
        end_time = time()
        delta = 0.00625

        for text in texts:
            begin_time = time()
            stdscr.erase()

            mode, rendertext = text.split('\n', 1)
            stdscr.bkgd(' ', curses.color_pair(getDisplayMode(mode)))

            stdscr.addstr(rendertext)
            sleep(1/fps - (begin_time - end_time))
            stdscr.refresh()
            end_time = time()


if __name__ == "__main__":
    curses.wrapper(main)
