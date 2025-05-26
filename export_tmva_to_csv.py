import uproot
import pandas as pd

# Open TMVA ROOT file
file = uproot.open("TMVA_diphoton.root")

# Access trees
train_tree = file["dataset/TrainTree"]
test_tree = file["dataset/TestTree"]

# Convert to pandas DataFrames
train_df = train_tree.arrays(library="pd")
test_df = test_tree.arrays(library="pd")

# Save to CSV
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Exported train_data.csv and test_data.csv successfully.")
