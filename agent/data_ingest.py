from __future__ import annotations
import pandas as pd
from .utils import basic_clean

def load_any(path_or_buffer) -> pd.DataFrame:
    name = getattr(path_or_buffer, 'name', '')
    if name.lower().endswith('.xlsx') or name.lower().endswith('.xls'):
        df = pd.read_excel(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer)
    return basic_clean(df)
