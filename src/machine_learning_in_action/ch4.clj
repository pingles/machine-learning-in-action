(ns machine-learning-in-action.ch4
  (:use [incanter.io :only (read-dataset)]
        [incanter.core :only (log to-matrix matrix dim plus sum div mult)]
        [clojure.set :only (intersection union difference)]
        [clojure.contrib.string :only (lower-case split)])
  (:import [java.io File]))

(def posting-list [["my", "dog", "has", "flea", "problems", "help", "please"]
                   ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"]
                   ["my", "dalmation", "is", "so", "cute", "I", "love", "him"]
                   ["stop", "posting", "stupid", "worthless", "garbage"]
                   ["mr", "licks", "ate", "my", "steak", "how", "to", "stop","him"]
                   ["quit", "buying", "worthless", "dog", "food", "stupid"]])

(def classes [0 1 0 1 0 1]) ; 1 is abusive/negative

(defn tokenise
  "Crudely returns tokens for input string. Kind of assumes input is english :)"
  [s]
  (split #"[^\w]+" s))

(defn vocab
  "Return vocab for a sequence of strings"
  [xs]
  (set (map lower-case xs)))

(defn vocab-list
  "Builds a vocab list across coll of tokenised documents. Assumes each doc is represented by a string"
  [coll]
  (apply sorted-set (apply union (map vocab coll))))

(defn vocab-vector
  "Returns a vector counting the words in xs that exist in the vocab. Words are returned in the order listed in vocab"
  [vocab xs]
  (let [occurrences (apply hash-map (flatten (map (fn [[k v]] [k (count v)])
                                                  (group-by identity xs))))]
    (map #(or (get occurrences %)
              0)
         vocab)))

(defn train-nb0
  [train-matrix categories]
  (let [[docs tokens] (dim train-matrix)
        p-abusive (/ (sum categories)
                     docs)
        category-and-doc (partition 2 (interleave categories train-matrix))
        ;; this below selects all rows relevant to the p0 category
        res0 (map last (filter #(= 0 (first %)) category-and-doc))
        res1 (map last (filter #(= 1 (first %)) category-and-doc))
        p0-num (reduce plus (conj (matrix res0) (repeat tokens 1)))
        p1-num (reduce plus (conj (matrix res1) (repeat tokens 1)))
        p0-denom (reduce + 2 (map sum res0))
        p1-denom (reduce + 2 (map sum res1))]
    {:abusive p-abusive
     :categories {0 (log (div p0-num p0-denom))
                  1 (log (div p1-num p1-denom))}}))

(defn classify-nb
  [vec p0vec p1vec pclass1]
  (let [p0 (+ (sum (mult (matrix [vec]) p0vec))
              (log pclass1))
        p1 (+ (sum (mult (matrix [vec]) p1vec))
              (log (- 1 pclass1)))]
    (if (> p1 p0)
      {:0 p0 :1 p1 :result 1}
      {:0 p0 :1 p1 :result 0})))


;; (def all-vocab (vocab-list posting-list))
;; (def our-matrix (matrix (map (partial vocab-vector all-vocab) posting-list)))
;; (def m (train-nb0 our-matrix classes))
;; (def i-bad (vocab-vector all-vocab (nth posting-list 1)))
;; (classify-nb i-bad (get-in m [:categories 0]) (get-in m [:categories 1]) 0.5)
;; 
;; {:0 -23.959247129717546, :1 -18.405536949199654, :result 1}

