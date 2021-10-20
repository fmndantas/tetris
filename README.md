# tetris

Classic game made with python

![](demo_tetris.gif)

## Installation

### Linux

One of the dependencies is the keyboard package, that
will run properly only if it was installed as root. 
To install from source, proceed with

```
git clone https://github.com/fmndantas/tetris.git
cd tetris
sudo pip3 install .
```

## Run game

### Linux

With ``sudo tetris`` you run the game. The movements are performed with keyboard, through
* ``a``: move shape to left
* ``d``: move shape to right
* ``s``: move shape down
* ``r``: rotate current shape

## An important observation
At the time of Tetris development, I had no knowledge about Desing Patterns and other advanced topics related to the object-oriented paradigm. Due to my ignorance, the code is high-coupled and certainly isn't easily extensible. I have no plans for refactoring any time soon.
