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
  (let [[rows cols] (dim data)
        col-range (range cols)
        mins (trans (map #(repeat rows (apply min (sel data :cols %)))
                         col-range))
        ranges (trans (map #(repeat rows (- (apply max (sel data :cols %))
                                            (apply min (sel data :cols %))))
                           col-range))]
    (div (minus data mins) ranges)))


(defn sorted-indexes
  "Sorts coll and returns indexes"
  [coll]
  (map first
       (sort-by last (partition 2
                                (interleave (range (count coll))
                                            coll)))))

(defn- tally-map
  [coll]
  (reduce (fn [h n]
            (assoc h n (inc (h n 0))))
          {} coll))

(defn mode
  [coll]
  (let [amap (tally-map aseq)
        mx (apply max (vals amap))
        modes (map key (filter #(= mx (val %)) amap))
        c (count modes)]
    (cond
      (= c 1) (first modes)
      (= c (count amap)) nil
      :default modes)))

(defn- euclidean-distance
  [xs m]
  (let [diff-matrix (minus (matrix (repeat (first (dim m)) xs)) m)
        sq-diff (sq diff-matrix)
        sq-distances (map sum sq-diff)
        distances (sqrt sq-distances)]
    distances))

(defn knn-classify
  "Classifies vector xs against nearest k neighbours in matrix m of known vectors labelled by labels"
  [xs k m labels]
  (let [sorted-distances (sorted-indexes (euclidean-distance xs m))
        sorted-labels (take k (map (partial nth labels)
                                   sorted-distances))
        likely-labels (mode sorted-labels)]
    (if (seq? likely-labels)
      (first likely-labels)
      likely-labels)))

(defn test-classifier
  [mdata]
  (let [labels (last (trans mdata))
        trained (normalise (sel mdata
                                :cols (range 3)
                                :rows (range 0 990)))
        test (normalise (sel mdata
                             :cols (range 3)
                             :rows (range 991 1000)))
        actual (drop 991 labels)]
    (partition 2 (interleave (map #(knn-classify % 5 trained labels)
                                  test)
                             actual))))
