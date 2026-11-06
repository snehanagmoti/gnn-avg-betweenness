while pgrep -f "queue1.sh" > /dev/null; do sleep 10; done
python gnn3.py N07_wtop2_extra300 '{"weight_top":2,"n_extra":300,"graphs_per_epoch":40}'
python gnn3.py N08_wtop2_extra300_long '{"weight_top":2,"n_extra":300,"graphs_per_epoch":40,"patience":40,"epochs":250}'
python gbm.py G18_reg_wt_anomfeat '{"obj":"regression","weight_top":2,"blocks":["xn","xe","xp","xa"]}'
