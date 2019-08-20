import sys

try:
    import pupil_parse.preprocess_utils
    import pupil_parse.summary_utils
    print('imports are successful!')
    print(help(pupil_parse.preprocess_utils))
    print(help(pupil_parse.summary_utils))
except:
    print('imports didn\'t work.')
