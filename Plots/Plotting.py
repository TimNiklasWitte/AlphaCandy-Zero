import sys
sys.path.append("../")

import pandas as pd
import numpy as np
from tbparse import SummaryReader

log_dir = "../test_logs"
reader = SummaryReader(log_dir)
df = reader.tensors

print(df)

