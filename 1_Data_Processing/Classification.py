import pandas as pd

# **1.contig_info.csv**
file_path = 'contig_info.csv'
contig_data = pd.read_csv(file_path)
new_column_names = [
    "Contig name",
    "Number of restriction sites",
    "Contig length",
    "Contig coverage",
    "Hi-C contacts mapped to the same contigs"
]
contig_data.columns = new_column_names

# **2. binned_contig.txt**
binned_file_path = 'binned_contig.txt'
binned_contig_data = pd.read_csv(binned_file_path, sep='\t', header=None)
binned_contig_data.columns = ["Contig name", "Bin"]

# **3. DemoVir_assignments.txt**
demo_vir_file_path = 'DemoVir_assignments.txt'
demo_vir_data = pd.read_csv(demo_vir_file_path, sep='\t')
demo_vir_data["Sequence_ID"] = demo_vir_data["Sequence_ID"].apply(lambda x: '_'.join(x.split('_')[:2]))
demo_vir_data["Order"] = "o_" + demo_vir_data["Order"].astype(str)
demo_vir_data["Family"] = "f_" + demo_vir_data["Family"].astype(str)
demo_vir_data["classification"] = demo_vir_data["Order"] + ";" + demo_vir_data["Family"]
demo_vir_data.rename(columns={"Sequence_ID": "Contig name"}, inplace=True)
demo_vir_data = demo_vir_data[["Contig name", "classification"]]

# **4. ppr_meta_result.csv**
ppr_meta_file_path = 'ppr_meta_result.csv'
ppr_meta_data = pd.read_csv(ppr_meta_file_path)

def classify(row):
    classification = []
    if row['phage_score'] > 0.5:
        classification ="phage"
    elif row['chromosome_score'] > 0.5:
        classification = "chromosome"
    elif row['plasmid_score'] > 0.5:
        classification = "plasmid"
    else:
        classification = "Unmapped"
    return classification

ppr_meta_data["type"] = ppr_meta_data.apply(classify, axis=1)
ppr_meta_data.rename(columns={"Header": "Contig name"}, inplace=True)
ppr_meta_data = ppr_meta_data[["Contig name", "type"]]

# **5. query_plasmid.txt**
query_plasmid_file_path = 'query_plasmid.txt'
query_plasmid_data = pd.read_csv(query_plasmid_file_path, sep='\t', header=None)
column_names = ["Contig name", "classification", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8", "Column9", "Column10", "Column11", "Column12"]
query_plasmid_data.columns = column_names
query_plasmid_data = query_plasmid_data[["Contig name", "classification"]]

# **6. metacc.gtdbtk.bac120.summary.tsv**
metacc_file_path = 'metacc.gtdbtk.bac120.summary.tsv'
metacc_data = pd.read_csv(metacc_file_path, sep='\t')
metacc_data.rename(columns={"user_genome": "Bin"}, inplace=True)
metacc_data = metacc_data.iloc[:, :2]

# **7. Getting classifications**
contig_data_with_bin = pd.merge(contig_data, binned_contig_data, on="Contig name", how="left")
contig_data_with_classification = pd.merge(contig_data_with_bin, metacc_data, on="Bin", how="left")

contig_data = pd.merge(contig_data_with_classification, ppr_meta_data, on="Contig name", how="left")

combined_data = pd.merge(contig_data, query_plasmid_data, on="Contig name", how="left", suffixes=("", "_plasmid"))
combined_data = pd.merge(combined_data, demo_vir_data, on="Contig name", how="left", suffixes=("", "_vir"))

def choose_classification(row):
    if row['type'] == "phage":
        return row['classification_vir']
    elif row['type'] == "plasmid":
        return row['classification_plasmid']
    elif row['type'] == "Unmapped":
        return "Unmapped"
    else:
        return row['classification']

combined_data['classification'] = combined_data.apply(choose_classification, axis=1)

combined_data['classification'] = combined_data['classification'].apply(
    lambda x: "Unmapped" if pd.isna(x) or "Unclassified" in x else x
)

combined_data = combined_data.drop(columns=['Bin', 'type', 'classification_plasmid', 'classification_vir'])

combined_data_file_path = 'contig_info_final.csv'
combined_data.to_csv(combined_data_file_path, index=False)