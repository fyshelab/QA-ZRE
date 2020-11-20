#%% 
# Preprocess Dreamscape Dataset
# VS Code Notebook

#%%
import pandas as pd

from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
#%%
passages_df = pd.read_csv(
    repo_root/'data/dreamscape/passages.csv',
    encoding='utf8', # Maybe , encoding="latin-1"
    quotechar="\"", 
    escapechar='\\', 
    quoting=0
)

# %%
