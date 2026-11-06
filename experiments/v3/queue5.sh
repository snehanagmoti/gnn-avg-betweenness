python gnn3.py N13_final_5seeds '{"weight_top":2,"hidden":64,"n_extra":300,"graphs_per_epoch":40,"edge_blocks":["xe","xp","xh"],"seeds":[0,1,2,3,4]}'
python ens.py N13_final_5seeds
python gen.py N13_final_5seeds
