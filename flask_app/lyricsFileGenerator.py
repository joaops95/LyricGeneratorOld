# some_file.py
import sys
import re
import pronouncing
from big_phoney import BigPhoney
import collections
import random
from PyLyrics import *


#  insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './PyLyrics-master/PyLirics')

def appendRymesinText(text, file_to_save):
    phoney = BigPhoney()
    lines = text.split("\n")
    counterRymesLines = 0
    checkRymesinLine = 5
    phoneyinline = []
    addedflag = 0
    scores = []
    linestoappend = []
    for line in lines:
        line = phoney.apply_preprocessors(line)
        # print(line)
        words = line.split(" ")
        if(counterRymesLines > checkRymesinLine):
            scores = []
            addedflag = 0
            for i in range(0, checkRymesinLine):
                sylables = phoneyinline[0][0][0]
                score = 0
                for item in sylables:
                    if(i < checkRymesinLine-1):
                        factor = phoneyinline[i+1][i+1][1]
                        for nextlinephoney in phoneyinline[i+1][i+1][0]:
                            if(item == nextlinephoney):
                                score += 1
                scorefactor = score/factor
                scores.append(scorefactor)
                score = 0 
                if(scorefactor > random.uniform(1.45, 2.2) and scorefactor < random.uniform(4.5, 6.0)):
                    if(addedflag == 0):
                        linestoappend.append(phoneyinline[0][0][2])
                    addedflag = 1
                    linestoappend.append(phoneyinline[i+1][i+1][2])
            counterRymesLines = 0
        addedflag = 0
        print(scores)

        if(counterRymesLines == 0):
            phoneyinline = []
        lastline = []

        for idx, word in enumerate(words):
            phoneyword = phoney.phonize(word).split(" ")
            numberofwords = idx+1
            for phoneysilable in phoneyword:
                lastline.append(phoneysilable)
        if(numberofwords > 2):
            element = (lastline,numberofwords,line)
            phoneyinline.append({counterRymesLines:element})
            counterRymesLines += 1
    unique_lines = set(linestoappend)
    text = ""
    print(len(linestoappend))
    for line in unique_lines:
        text += line + "\n"
    file1 = open(file_to_save,"a")
    file1.write(text)
    file1.close()


# appendRymesinText(rhymes, 'musiclyrics.txt')

artists = [
    'Drake', 'Eminem', 'J. Cole', 'Kendrick Lamar', 'Da Baby', '2Pac', 'Logic', 'Juice Wrld', 'Travis Scott'
]
    
def createDataSet(artists):
    for artist in artists:
        albums = PyLyrics.getAlbums(singer=artist)[-4:-1]
        for album in albums:
            tracks = PyLyrics.getTracks(album)
            for track in tracks:
                try:
                    print(artist)
                    print(album)
                    print(track)
                    lyrics = track.getLyrics()
                    lyrics = re.sub("[\<\{\(\[].*?[\)\}\>\]]", "", lyrics)
                    appendRymesinText(lyrics, 'musiclyrics.txt')

                except: continue



createDataSet(artists)

