import pandas as pd

# Load the test.csv file
test_df = pd.read_csv('./dataset/test.csv')

# Load the test_out.csv file
test_out_df = pd.read_csv('test_out.csv')

# Ensure the test_out_df has the same number of rows as test_df
if len(test_df) != len(test_out_df):
    raise ValueError("The number of rows in test.csv and test_out.csv do not match.")

# Add the index from test_df to test_out_df
test_out_df['index'] = test_df.index

# Save the updated test_out_df to test_out.csv
test_out_df.to_csv('test_out.csv', index=False)

print("Index column updated successfully.")