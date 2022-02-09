home_path = '/home/lizhen/data/'
#threshold_xi = 5e-6
#threshold_gamma = 0.6
threshold_xi = 10e-6
threshold_gamma = 0.4
information_flow_alpha = 0.4
information_flow_beta = 5
n_percentile = 10
n_percentile_words = 20
word_context_max_layers = 4
word_contribution_max_layers = 6
stop_words = ['ziff', 'woe', 'lt', 'gt', 'quot', 'rsquo', 'quickinfo', 'quote', 'arial', 'com', 'href', 'aspx', 'helvetica', 'serif', 'www', 'fullquote', 'ap', 'afp', 'sans', 'font']# 'alligators', 'crocodilians', 'photoshop', 'llinas', 'brockton', 'puncha', 'manhood']

project_name = 'DeepNLPVis'

dataset_name = 'sst2_lstm'
model_name = 'lstm'
n_samples = 4000
'''
dataset_name = 'sst2_10k'
model_name = 'bert'
n_samples = 4000

dataset_name = 'sst2_lstm'
model_name = 'lstm'
n_samples = 4000

dataset_name = 'agnews'
model_name = 'bert'
n_samples = 2000
'''


max_keyword_num = 500
flow_max_degree = 3
flow_max_cross = 4
flow_max_phrase_len = 10
flow_n_lines = 25
flow_expected_min_grids = 60
flow_expected_max_grids = 80
flow_word_per_grids = 4
flow_deep_layer = 6
context_max_n_clusters = 7
context_min_n_clusters = 2
context_distance_threshold = 3
context_distance_step = 1.1

frequent_phrase_step = 1.2