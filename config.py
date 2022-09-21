ga = {
    "population": 100,
    "n_gen":400,
    "seed": 873,
    "mut_rate": 0.4,
    "cross_rate": 1
} 

model = {
    "map_size": 40,#50,
    "min_len": 8,#3,  # minimal possible distance in meters
    "max_len": 15,#12,#15,  # maximal possible disance to go straight in meters
    "min_pos": 1,  # minimal possible 
    "max_pos": 38, #49 29,  # maximal possible position in meters
    "len_step": 1,
    "pos_step": 1,
    "max_changes": 35,

}

files = {
    "logs_path": ".\\2022-09-18_logs\\",
    "model_path": ".\\2022-09-18_models\\",
    "img_path": ".\\2022-09-18_images\\",
}