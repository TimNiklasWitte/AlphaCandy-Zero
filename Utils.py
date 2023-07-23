from Config import *
from StateToImageConverter import *

def preprocess_single_raw_state(state, stateToImageConverter):

    state_img = stateToImageConverter(state)
    # state_img = state_img.astype("float32")
    # state_img = (state_img / 128.) - 1.

    return state_img