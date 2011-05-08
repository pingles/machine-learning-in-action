(ns machine-learning-in-action.ch2
  (:use [incanter.io :only (read-dataset)]
        [incanter.core :only (to-matrix sel)]
        [incanter.charts :only (scatter-plot)]
        [incanter.stats :only (correlation)]))

(def data (read-dataset "./data/ch1/datingTestSet2.txt" :delim \tab))
(def mdata (to-matrix data))

;; (view (scatter-plot :col3 :col1 :data data))
;; (view (scatter-plot :col0 :col1 :group-by :col3 :data data))
;; (def c (correlation mdata))

(defn normalise
  "Normalises values in matrix m to between 0 and 1"
  [m]
  (let [[rows cols] (dim m)
        col-range (range cols)]
    (div (minus m
                (trans (map #(repeat rows (apply min (sel m :cols %)))
                            col-range)))
         (trans (map #(repeat rows (- (apply max (sel m :cols %))
                                      (apply min (sel m :cols %))))
                     col-range)))))

(defn sorted-indexes
  "Sorts coll and returns indexes"
  [coll]
  (map first
       (sort-by last (partition 2
                                (interleave (range (count coll))
                                            coll)))))

(defn mode
  [coll]
  (let [tallies (reduce (fn [h n]
                          (assoc h n (inc (h n 0))))
                        {} coll)
        mx (apply max (vals tallies))
        modes (map key (filter #(= mx (val %)) tallies))
        c (count modes)]
    (cond (= c 1) (first modes)
          (= c (count tallies)) nil
          :default modes)))

(defn euclidean-distance
  "Calculates euclidean distance between vector xs and all vectors in matrix m."
  [xs m]
  (let [xs-matrix (matrix (repeat (first (dim m))
                                  xs))]
    (sqrt (map sum
               (sq (minus xs-matrix
                          m))))))

(defn knn-classify
  "Classifies vector xs against nearest k neighbours in matrix m of known vectors labelled by labels"
  [xs k m labels]
  (let [sorted-labels (take k (map (partial nth labels)
                                   (sorted-indexes (euclidean-distance xs m))))
        category (mode sorted-labels)]
    (if (seq? category)
      (first category)
      category)))

(defn test-classifier
  [mdata]
  (let [labels (last (trans mdata))
        trained (normalise (sel mdata
                                :cols (range 3)
                                :rows (range 0 980)))
        test (normalise (sel mdata
                             :cols (range 3)
                             :rows (range 981 1000)))
        actual (drop 981 labels)]
    (partition 2 (interleave (map #(knn-classify % 5 trained labels)
                                  test)
                             actual))))
