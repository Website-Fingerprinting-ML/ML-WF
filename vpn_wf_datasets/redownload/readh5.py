import pandas as pd
df = pd.read_hdf("VNAT_Dataframe_release_1.h5", key="data")

print(df.head())      # first 5 rows
print(df.columns)     # column names
print(df.info())      # structure

for name in df["file_names"]:
    print(name)

# with pd.HDFStore("VNAT_Dataframe_release_1.h5") as store:
#     print(store.keys())
