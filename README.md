# video2asciis

![sample gif](https://github.com/private-yusuke/video2asciis/blob/master/sample.gif?raw=true)

(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org

---

video2asciis lets you create old-school style videos like shown above.

---

## Requirements

```sh
$ pip3 install python-opencv numpy tqdm pymediainfo ffmpeg
```

## Usage

```sh
$ ./v2t2v.sh <input> <output> <font>
```

This handy command generates the converted video in the same resolution as the original.

You can modify some parameters by directly using `v2t.py` and `t2v.py`. For details, refer to the output of `$ python3 v2t.py -h` and `$ python3 t2v.py -h`.

*Note*: If the given file doesn't have an audio track, then this script may throw an error. You can manually try `t2v.py` to obtain what you want.

## Play your video in your terminal

You can play your converted texts in your terminal in real time.

1. `$ python3 v2t.py <input> out.txt`
2. `$ python3 tshow.py out.txt`
3. Enjoy!
    * if you can't play the file, it's possibly because your terminal is too small to display the entire text. Try reducing the font size of the terminal, and enlarge the terminal.

## Tips

* in `v2t.py`, the threshold between white and black is defined as `90`. You may have to change this value by passing `-t <value>` option, since this value is critical to generate the appropriate results.
   * When there are many bright scenes overall, you may choose `95~110`.
* `v2t.py` accepts an option `-s <size>`. This program divides the input frame into pieces that are shaped as `size x (size/2)`. Each piece are then substituted with one character.
    
    The bigger size you specify, the result is rough and the conversion is faster.
* `v2t.py` also supports the custom settings of output frame size.

---

Need information in Japanese? If so, you may want this: https://scrapbox.io/public-yusuke/%E6%96%87%E5%AD%97%E5%88%97%E3%81%A7%E5%8B%95%E7%94%BB%E3%82%92%E8%A1%A8%E7%8F%BE%E3%81%99%E3%82%8B
