(ns machine-learning-in-action.ch4
  (:use [incanter.io :only (read-dataset)]
        [incanter.core :only (to-matrix)]
        [clojure.set :only (intersection union)]
        [clojure.contrib.string :only (lower-case split)]))

(def data (read-dataset "./data/ch4/"))

(def posting-list [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage']
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop','him']
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']])
(def classes [0 1 0 1 0 1]) ; 1 is abusive/negative

(defn tokenise
  "Crudely returns tokens for input string. Kind of assumes input is english :)"
  [s]
  (split #"[^\w]+" s))

(defn vocab
  [s]
  (set (map lower-case (tokenize s))))

(defn vocab-list
  "Builds a vocab list across coll of documents. Assumes each doc is represented by a string"
  [coll]
  (apply union (map vocab coll)))


