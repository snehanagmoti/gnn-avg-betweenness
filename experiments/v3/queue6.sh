python gnn3.py N14_mixed_types '{"weight_top":2,"hidden":64,"n_extra":150,"graphs_per_epoch":40,"edge_blocks":["xe","xp","xh"],"extra_sets":[["er",60,200],["ktree",60,200],["random_tree",60,200],["ba",20,400]]}'
python gen.py N14_mixed_types
python ref_gen.py
python timing.py
