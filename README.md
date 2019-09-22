# Word Analogizer
<p>
Paris is to France as Berlin is to ?
</p>
<p>
Car is to road as train is to ?
</p>
<p>
King is to queen as husband is to ?
</p>
<p> These all seem like really easy analogies, and they are! Errr... for humans, at least. For machines, not so much. That's why I tried creating a little program capable of taking advantage of word vectorization and Wikipedia's entire corpus to predict the answer to analogies like those above!
</p>
<p> This program has around a 19% accuracy rate, which might seem terrible but hear me out: the algorithm is not choosing from a pool of words (like multiple choice); it's actually supplying its own output (like short answer on tests). From past experience, I can emphatically say randomly guessing on short answers does NOT give correct solutions 1/5 of the time.
</p>

How to have fun with Word Analogizer
  1) Clone this repo
  2) Download pip
  3) Navigate to the `word-analogizer` folder and type `pip install -r requirements.txt`  
  4) Run `analogy.py`
  5) Have fun!
