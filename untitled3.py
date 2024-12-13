import pandas as pd

# Load the uploaded CSV files
main_table_path = 'binned_contig.csv'
taxonomy_info_path = 'taxonomy_information.csv'

# Read the CSV files into DataFrames
binned_contig_df = pd.read_csv(main_table_path)
taxonomy_info_df = pd.read_csv(taxonomy_info_path)

# Merge the tables based on "Contig index" from the main table and "Bin index" from the taxonomy table
merged_df = binned_contig_df.merge(taxonomy_info_df, left_on='Contig index', right_on='Bin index', how='left')

# Modify the DataFrame based on the given conditions
# If 'Category' == 'virus', add 'v' in front of the 'Bin index_x' value
merged_df.loc[merged_df['Category'] == 'virus', 'Bin index_x'] = 'v' + merged_df['Bin index_x']

# If 'Category' == 'plasmid', delete the row

virus_df = merged_df[merged_df['Category'] == 'virus']

filtered_df = merged_df[merged_df['Category'] != 'plasmid']
filtered_df = filtered_df[filtered_df['Category'] != 'virus']

# Rank the 'Bin index_x' values based on their frequency of appearance
ranked_bin_counts = virus_df['Bin index_x'].value_counts().reset_index()
ranked_bin_counts.columns = ['Bin index_x', 'Count']

# Create a ranking column
ranked_bin_counts['Rank'] = range(1, len(ranked_bin_counts) + 1)
rank_mapping = ranked_bin_counts.set_index('Bin index_x')['Rank'].to_dict()
virus_df['Bin index_x'] = virus_df['Bin index_x'].map(lambda x: f"vMAG_{rank_mapping[x]}")

grouped_virus_df = virus_df.groupby('Bin index_x').agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None
).reset_index()






combined_df = pd.concat([filtered_df, virus_df])

# Save the merged DataFrame to a CSV file
output_file_path = 'grouped_virus_df.csv'
grouped_virus_df.to_csv(output_file_path, index=False)
