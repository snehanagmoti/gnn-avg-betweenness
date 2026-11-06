while pgrep -f "gbm.py G17" > /dev/null; do sleep 10; done
python gnn3.py N01_reg_xe_xp '{}'
python gnn3.py N02_mlp_only '{"no_gnn":true}'
python gnn3.py N03_reg_wtop2 '{"weight_top":2}'
python gnn3.py N04_reg_triplet '{"loss":"reg+triplet"}'
python gnn3.py N05_reg_xe_only '{"edge_blocks":["xe"]}'
python gnn3.py N06_triplet_xe_xp '{"loss":"triplet"}'
