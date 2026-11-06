while pgrep -f "queue2.sh" > /dev/null; do sleep 10; done
python gnn3.py N09_wtop2_hist '{"weight_top":2,"edge_blocks":["xe","xp","xh"]}'
python gnn3.py N10_wtop2_h64 '{"weight_top":2,"hidden":64}'
python gnn3.py N06_triplet_xe_xp '{"loss":"triplet"}'
