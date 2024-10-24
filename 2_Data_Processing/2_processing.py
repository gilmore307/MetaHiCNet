import pandas as pd
from plsdbapi import query

# **0. Helper Functions**
def adjust_taxonomy(row):
    last_non_blank = ""
    
    for tier in taxonomy_columns:
        row[tier] = str(row[tier]) if pd.notna(row[tier]) else ""

    if row['Type'] != 'unmapped':
        for tier in taxonomy_columns:
            if row[tier]:
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"
    else:
        for tier in taxonomy_columns:
            row[tier] = "unmapped"

    if row['Type'] == 'phage':
        row['Domain'] = 'Virus'
        row['Phylum'] = 'Virus'
        row['Class'] = 'Virus'
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_v'
        row['Contig'] = row['Contig'] + "_v"
        row['Bin'] = row['Bin'] + "_v"

    if row['Type'] == 'plasmid':
        # Add suffix '_p' to all taxonomy levels for plasmids
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_p'
        row['Contig'] = row['Contig'] + "_p"
        row['Bin'] = row['Bin'] + "_p"

    for tier, prefix in prefixes.items():
        row[tier] = f"{prefix}{row[tier]}" if row[tier] else "N/A"

    return row

# **1. contig_information.csv**
contig_data_path = 'input/contig_information.csv'
contig_data = pd.read_csv(contig_data_path)

# **2. binning_information.csv**
binning_data_path = 'input/binning_information.csv'
binning_data = pd.read_csv(binning_data_path)

# **3. taxonomy.csv**
taxonomy_data_path = 'input/taxonomy.csv'
taxonomy_data = pd.read_csv(taxonomy_data_path)

plasmid_ids = taxonomy_data['Plasmid ID'].dropna().unique().tolist()
plasmid_classification_df = query.query_plasmid_id(plasmid_ids)[[
    'NUCCORE_ACC', 
    'TAXONOMY_superkingdom', 
    'TAXONOMY_phylum', 
    'TAXONOMY_class', 
    'TAXONOMY_order', 
    'TAXONOMY_family', 
    'TAXONOMY_genus', 
    'TAXONOMY_species'
]]

plasmid_classification_df.rename(columns={
    'NUCCORE_ACC': 'Plasmid ID',
    'TAXONOMY_superkingdom': 'Kingdom',
    'TAXONOMY_phylum': 'Phylum',
    'TAXONOMY_class': 'Class',
    'TAXONOMY_order': 'Order',
    'TAXONOMY_family': 'Family',
    'TAXONOMY_genus': 'Genus',
    'TAXONOMY_species': 'Species'
}, inplace=True)

prefixes = {
    'Domain': 'd_',
    'Kingdom': 'k_',
    'Phylum': 'p_',
    'Class': 'c_',
    'Order': 'o_',
    'Family': 'f_',
    'Genus': 'g_',
    'Species': 's_'
}

plasmid_classification_df = plasmid_classification_df.replace(r"\s*\(.*\)", "", regex=True)
plasmid_classification_df['Domain'] = plasmid_classification_df['Kingdom']

taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

taxonomy_data = taxonomy_data.merge(
    plasmid_classification_df[['Plasmid ID'] + taxonomy_columns],
    on='Plasmid ID',
    how='left',
    suffixes=('', '_new')
)

for column in taxonomy_columns:
    taxonomy_data[column] = taxonomy_data[column + '_new'].combine_first(taxonomy_data[column])

taxonomy_data = taxonomy_data.drop(columns=['Plasmid ID'] + [col + '_new' for col in taxonomy_columns])

# **4. contig_info_final**
combined_data = pd.merge(contig_data, binning_data, on="Contig", how="left")
combined_data = pd.merge(combined_data, taxonomy_data, on="Bin", how="left")

combined_data = combined_data.apply(adjust_taxonomy, axis=1)
combined_data['Bin'] = combined_data['Bin'].fillna('Unbinned MAG')


# **6. Save**
combined_data_path = 'output/contig_info_complete.csv'
combined_data.to_csv(combined_data_path, index=False)