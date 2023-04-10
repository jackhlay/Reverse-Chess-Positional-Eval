import pandas as pd
import random

pos = pd.read_csv('Positions.csv')

shuff = pos.sample(frac=1, random_state=890732).reset_index(drop=True)