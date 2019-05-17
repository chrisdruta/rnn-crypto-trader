#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime

def daily_parse(filename):
    """
    Opens and parses the given filename inside ./data/ directory

    Args:
        filename: String containing a csv filename

    Returns:
        Nothing; writes parsed csv to ./data/ directory with same filename
    """
    data = pd.read_csv(f"./data/{filename}.csv", header=0).values
    new = []
    for row in data:
        nrow = []
        nrow.append(datetime.strptime(row[0], "%b %d, %Y").__str__()[:10])
        nrow.extend(row[1:-2])
        nrow.append(int(row[-1].replace(',','')))
        new.append(nrow)
    pd.DataFrame(new[::-1]).to_csv(f"./data/{filename}_parsed.csv", index=False, header=False, mode='w')

daily_parse('bitcoin')