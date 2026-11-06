python gnn3.py N11_h64_extra300 '{"weight_top":2,"hidden":64,"n_extra":300,"graphs_per_epoch":40}'
python gnn3.py N12_h64_extra300_hist '{"weight_top":2,"hidden":64,"n_extra":300,"graphs_per_epoch":40,"edge_blocks":["xe","xp","xh"]}'
python ens.py N10_wtop2_h64
python ens.py N10_wtop2_h64 N07_wtop2_extra300 N09_wtop2_hist
