from pupil_parse.edf2pd import * 


subj_id = 786
session_n = 1

subj_data_file, reward_code = find_data_files(subj_id, session_n,
reward_task=1, lum_task=0)

samples, events, messages = read_edf(subj_data_file)

samples, events, messages = clean_df(samples, events, messages)

samples, events, messages = extract_experimental_data(samples, events, messages)

samples, events, messages = define_relative_time(samples, events, messages)

hdf = save_hdf5(samples, events, messages,
subj_id, session_n, reward_code=None, reward_task=1, lum_task=0)
