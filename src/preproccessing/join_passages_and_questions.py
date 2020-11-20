#%% 
# Preprocess Dreamscape Dataset
# VS Code Notebook

#%%
import pandas as pd

from io import StringIO
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
#%%
passages_df = pd.read_csv(
    repo_root/'data/dreamscape/passages.csv',
    encoding='ansi',
    quotechar="\"", 
    escapechar='\\', 
    quoting=0
)
passages_df = passages_df[["id", "genreId", "gradeId", "text"]].to_csv(repo_root/'data/clean/passages.csv',encoding='utf8')

# %%
