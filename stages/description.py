from dash import html
import dash_bootstrap_components as dbc

modal_body = dbc.ModalBody(
    html.Div(
        [
            html.H2("Section 1: Preparation", className="mb-3"),
            html.P("MetaHiCNet accommodates both new and returning users with tailored input requirements.", 
                   className="mb-4"),

            html.H4("For New Users:", className="mt-3"),
            html.Div([
                html.H5("1. Contig Information File", className="mt-3"),
                html.P([
                    "This file includes the following columns:",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Contig index’"), ", ",
                            html.Strong("‘Number of restriction sites’"), ", and ",
                            html.Strong("‘Contig length’"), " (required)."
                        ]),
                        html.Li([
                            html.Strong("‘Contig coverage’"), " (optional): If not provided, it will be estimated by dividing the diagonal value in the raw Hi-C contact matrix by the ‘Contig length’."
                        ])
                    ]),
                    "This file can be directly generated from common Meta Hi-C analysis pipelines, such as ",
                    html.Strong("MetaCC"), " and ", html.Strong("HiCBin"), "."
                ], className="mb-3"),

                html.H5("2. Hi-C Contact Matrix", className="mt-3"),
                html.P([
                    "The file can be provided in one of the following formats:",
                    html.Ul([
                        html.Li([
                            html.Strong(".txt or .csv format"), ": Should contain the columns ",
                            html.Strong("‘Contig_name1’"), ", ",
                            html.Strong("‘Contig_name2’"), ", and ",
                            html.Strong("‘Contacts’"), "."
                        ]),
                        html.Li([
                            html.Strong(".npz format"), ": Should be either a ",
                            html.Strong("NumPy dense matrix"), " or a ",
                            html.Strong("SciPy sparse matrix"), "."
                        ])
                    ]),
                    "This file can be directly generated from common Meta Hi-C analysis pipelines, such as MetaCC and HiCBin.",
                    html.Br(),
                    html.Strong("Note:"), " The row and column indices of the Hi-C Contact Matrix must match the row indices of the Contig Information File. ",
                ], className="mb-3"),

                html.H5("3. Binning Information File (Optional)", className="mt-3"),
                html.P([
                    "Skip this step if your goal is solely to normalize the Hi-C contact matrix.",
                    html.Br(),
                    "File format: ",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Contig index’"), " and ",
                            html.Strong("‘Bin index’"), " (specifying the bin to which each contig belongs)."
                        ])
                    ]),
                    "This file can be obtained from the binning results of ", html.Strong("Meta Hi-C analysis pipelines"), " or any other binners you select."
                ], className="mb-3"),

                html.H5("4. Taxonomy Information File (Optional)", className="mt-3"),
                html.P([
                    "Skip this step if your goal is solely to normalize the Hi-C contact matrix.",
                    html.Br(),
                    "File format:",
                    html.Ul([
                        html.Li([
                            html.Strong("‘Bin index’")
                        ]),
                        html.Li([
                            html.Strong("‘Category’"), ": The taxonomic category of each bin (",
                            html.Strong("‘chromosome’"), ", ",
                            html.Strong("‘virus’"), ", ",
                            html.Strong("‘plasmid’"), ", or ",
                            html.Strong("‘Unclassified’"), "). Unclassified bins can also be left blank."
                        ]),
                        html.Li([
                            "Additional ", html.Strong("Taxonomic Columns"), " for taxonomic information (e.g., family, genus, species)."
                        ])
                    ])
                ], className="mb-3"),
            ]),

            html.H4("For Returning Users:", className="mt-3"),
            html.P([
                "Returning users can upload compressed files generated in the ",
                html.Strong("Hi-C Contact Normalization Results"), " page to restore their progress.",
                html.Br(),
                html.Ul([
                    html.Li([
                        html.Strong("Unnormalized Data"), ": Upload the ",
                        html.Strong("unnormalized_information.7z"), " file to move directly to the normalization stage."
                    ]),
                    html.Li([
                        html.Strong("Normalized Data"), ": Upload the ",
                        html.Strong("normalized_information.7z"), " file to proceed directly to the visualization stage."
                    ])
                ])
            ], className="mb-0"),
            
            html.H4("For a Try-Out:", className="mt-3"),
            html.P([
                "Simply click the ", html.Strong("Load Example Files"), " button to automatically load pre-prepared files, allowing you to move directly to the next step without uploading any data."
            ], className="mb-3"),
            
            html.Hr(),
            
            # Section 2: Normalization
            html.H2("Section 2: Normalization", className="mt-5 mb-3"),
            html.P("MetaHiCNet provides existing normalization methods for Hi-C contact matrices. Select the method based on your dataset’s characteristics and analysis needs.", className="mb-4"),

            # Raw Method
            html.H4("1. Raw", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " The Raw method does not remove the effects of any factors from the input Hi-C contact matrix. Select the Raw method if your input Hi-C contact matrix has already been normalized or if normalization is unnecessary for your analysis."
            ]),
            html.P([
                html.Strong("How does it work?"), " It only denoises the data by removing values below a specified threshold percentile."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: A value between 0–100% (default: 5%). Higher values remove more contacts.")
                ])
            ]),

            # normCC Method
            html.H4("2. normCC", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " NormCC is a normalization module within the MetaCC framework that eliminates systematic biases from metagenomic Hi-C data, such as biases caused by contig length, restriction sites, and coverage."
            ]),
            html.P([
                html.Strong("How does it work?"), " NormCC employs a negative binomial regression model to adjust Hi-C contact data for systematic biases. Unlike other methods, it does not require pre-computed contig abundances to normalize the data. After bias correction, normalized contacts falling below a specified threshold percentile are classified as spurious and removed."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: A value between 0–100% (default: 5%). Higher values remove more contacts.")
                ])
            ]),

            # HiCzin Method
            html.H4("3. HiCzin", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " HiCzin is a normalization method designed specifically for metagenomic Hi-C data. It addresses both explicit biases (e.g., contig length, restriction site counts, and coverage) and implicit biases (e.g., unobserved interactions) using a zero-inflated negative binomial regression framework. "
            ]),
            html.P([
                html.Strong("How does it work?"), " HiCzin combines two components: a) Negative Binomial Regression: Models the raw metaHi-C count data while accounting for explicit biases; b) Zero-Inflated Component: Captures unobserved interactions caused by experimental noise. After bias elimination, normalized contacts falling below a specified threshold percentile are classified as spurious and removed. "
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: A value between 0–100% (default: 5%). Higher values remove more contacts.")
                ])
            ]),

            # bin3C Method
            html.H4("4. bin3C", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " Bin3C is a pipeline designed for genome binning using metagenomic Hi-C data. It includes a normalization module that removes experimental biases to ensure uniform signals across the Hi-C contact map, which is critical for accurate binning."
            ]),
            html.P([
                html.Strong("How does it work?"), " The normalization process involves two stages: a) Cut-Site Normalization: Raw Hi-C interaction counts between contigs are then adjusted by dividing each count by the product of the cut site counts for the interacting contigs. This step addresses biases introduced by variation in restriction site density; b) Bistochastic Matrix Balancing: The Knight-Ruiz algorithm is applied to the adjusted Hi-C contact map. This algorithm transforms the matrix into a form where rows and columns have uniform totals, correcting residual biases and ensuring consistent interaction signals across the dataset. After bias elimination, normalized contacts falling below a specified threshold percentile are classified as spurious and removed."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: A value between 0–100% (default: 5%). Higher values remove more contacts."),
                    html.Li("Max Iterations: Default 1000; specifies the maximum number of iterations for the Knight-Ruiz algorithm."),
                    html.Li("Tolerance: Default 1e-6; defines the convergence threshold for the Knight-Ruiz algorithm.")
                ])
            ]),

            # MetaTOR Method
            html.H4("5. MetaTOR", className="mt-3"),
            html.P([
                html.Strong("What is this method?"), " MetaTOR is a computational pipeline designed for metagenomic binning using Hi-C data. The normalization module in MetaTOR processes Hi-C interaction data to correct for biases introduced by variations in contig coverage, facilitating accurate genome binning."
            ]),
            html.P([
                html.Strong("How does it work?"), " Interaction counts between contigs are normalized by dividing the edge weight (contact score) by the geometric mean of the coverage of the two contigs. This ensures that interaction frequencies are not skewed by differences in sequencing depth across contigs."
            ]),
            html.P([
                html.Strong("Parameter Settings:"),
                html.Ul([
                    html.Li("Threshold: A value between 0–100% (default: 5%). Higher values remove more contacts.")
                ])
            ]),
            
            # Additional Options
            html.H4("6. Additional Options (Advanced Options to Process Data After Normalization)", className="mt-4"),
            html.P("The following options can be enabled to further process the data after normalization:", className="mb-3"),

            # Remove Unclassified Contigs
            html.H5("Remove Unclassified Contigs:", className="mt-3"),
            html.P([
                "Check this box to exclude contigs or bins that are not classified in any taxonomic levels. Enabling this option helps reduce the size of your dataset and speeds up processing."
            ], className="mb-3"),

            # Remove Host-Host Interactions
            html.H5("Remove Host-Host Interactions:", className="mt-3"),
            html.P([
                "Check this box to remove all interactions between contigs or bins labeled as chromosomes. Enabling this option can significantly reduce the amount of Hi-C contacts and accelerate processing."
            ], className="mb-3"),

            # Important Note
            html.P([
                html.Strong("Important Note:"), " Please do not enable the Remove Unclassified Contigs and Remove Host-Host Interactions options if you have not provided the Binning Information File and Taxonomy Information File during the data upload process. These files are required to process these options correctly."
            ], className="mt-3"),
            
            html.H4("7. Normalization Results", className="mt-4"),
            html.H5("Figure: Relationship between raw interaction counts and the product of the number of restriction sites, length, and coverage between contig pairs.", className="mt-3"),  # Sub-header for Figure
            html.P([
                "The figure illustrates how raw Hi-C interaction counts are influenced by three bias factors: the number of restriction sites, contig length, and coverage between pairs of contigs.",
            ], className="mb-3"),
            html.P([
                html.Strong("X-axis:"), " Represents the product of one of the bias factors (restriction sites, length, or coverage).",
                html.Br(),
                html.Strong("Y-axis:"), " Represents the raw interaction counts."
            ], className="mb-3"),

            html.H5("Table: Pearson Correlation Coefficients (Absolute Value) Between Normalized Hi-C Contacts and the Product of Each of the Three Factors of Explicit Biases", className="mt-3"),  # Sub-header for Table
            html.P([
                html.Strong("Smaller values"), " are better, indicating that the normalization method has effectively removed the bias.",
                html.Br(),
                html.Strong("Larger values"), " suggest that the normalization method has not fully corrected for the corresponding bias.",
                html.Br(),
                html.Strong("Ideal Outcome: "), "Correlations close to 0 indicate successful bias removal."
            ], className="mb-3"),
            
            html.Hr(),
            
            # Section 3: Visualization
            html.H2("Section 3: Visualization", className="mt-5 mb-3"),
            html.P("Use the following visualization options to explore and analyze Hi-C interaction data in different ways.", className="mb-4"),

            # Switch Visualization Network
            html.H4("Switch Visualization Network", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Switch Visualization Network.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ),    
            html.P([
                "Click this button to switch from the current visualization to the normalization results view."
            ], className="mb-3"),

            # Reset Button
            html.H4("Reset Selection", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Reset Selection.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "Click this button to clear all selections and reset the visualization to Cross-Taxa Hi-C Interaction."
            ], className="mb-3"),

            # Tooltip Toggle Container
            html.H4("Enable or Disable Tooltips", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Enable or Disable Tooltips.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "Check this box to enable tooltips that provide contextual information about the components of this app."
            ], className="mb-3"),

            # Dropdowns
            html.H4("Visualization and Selection Dropdowns", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Visualization and Selection Dropdowns.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "Use these dropdown menus to explore different visualization modes or select annotations and bins for detailed analysis.",
                html.Br(),
                "The options selected here control the content displayed in the visualizations and tables.",
                html.Br(),
                "The dropdown offers three visualization options: ",
                html.Ul([
                    html.Li("Taxonomic Framework: Displays a hierarchical treemap showing how annotations are grouped and scaled by a selected metric, such as coverage or classification."),
                    html.Li("Cross-Taxa Hi-C Interaction: Focuses on interactions between annotations at a taxonomic level, shown as a Cytoscape graph and bar charts summarizing interaction metrics."),
                    html.Li("Cross-Bin Hi-C Interactions: Explores relationships between individual bins and their connections within the dataset, emphasizing specific bins of interest.")
                ])
            ], className="mb-3"),

            # Legend Container
            html.H4("Color Legend and Taxonomy Level Selector", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Color Legend and Taxonomy Level Selector.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "1. ", html.Strong("Color Legend:"), " Colors are consistently applied across the Cytoscape graph, bar chart, and tables to represent categories or annotations at the selected taxonomy level. Use the legend to identify categories by their assigned colors.",
                html.Ul([
                    html.Li("Viruses are reddish colors."),
                    html.Li("Plasmids are greenish colors."),
                    html.Li("Chromosomes are bluish colors.")
                ]),
                "2. ", html.Strong("Taxonomy Selector:"), " The taxonomy selector affects: ",
                html.Ul([
                    html.Li("Taxonomy Visualization: Adjusts how nodes are annotated and grouped in the hierarchy, such as by phylum or genus."),
                    html.Li("Bin Visualization: Influences how nodes (bins) are distributed in the network, grouping them according to the selected taxonomy level."),
                    html.Li("Contact Table: Changes the level of aggregation of the contact table by defining rows and columns based on the selected taxonomy level.")
                ])
            ], className="mb-3"),

            # Bar Chart Container
            html.H4("Bar Chart", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Bar Chart.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="30%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "1. ", html.Strong("Chart Types:"), " The bar chart can display the following types of charts: ",
                html.Ul([
                    html.Li("Fraction of Classified Bins by Taxonomic Ranks: Shows the percentage of bins classified at each taxonomic level (e.g., phylum, genus)."),
                    html.Li("Across Taxonomy Hi-C Contacts: Summarizes Hi-C contact strengths for each taxonomic annotation."),
                    html.Li("Hi-C Contacts with Selected Bin: Highlights the contact strengths between the selected bin and other bins.")
                ]),
                "2. ", html.Strong("Scroll Bar:"), " A horizontal scroll bar allows you to navigate through bars when there are too many to display at once."
            ], className="mb-3"),

            # Information Table Container
            html.H4("Information Table", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Information Table.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "1. ", html.Strong("Filter, Sort, and Search:"), " Use column headers to sort rows or apply filters to narrow down results. You can also use the search box in the headers to find specific bins or annotations quickly.",
                html.Br(),
                "2. ", html.Strong("Bin Selection:"), " Click a row to select a bin, updating the Cytoscape graph, bar chart, and other visualizations.",
                html.Br(),
                "3. ", html.Strong("Automatic Filtering:"), " The table updates dynamically based on selections. ",
                html.Ul([
                    html.Li("Taxa Selected: Shows bins within the selected taxa."),
                    html.Li("Bin Selected: Shows bins interacting with the selected bin.")
                ]),
                "4. ", html.Strong("Filter Checkbox:"), " Enabling 'Only show elements present in the diagram' checkbox displays only bins or annotations visible in the Cytoscape graph.",
                html.Br(),
                "5. ", html.Strong("Color Coding:"), " ",
                html.Ul([
                    html.Li("Index Column: Matches the node colors in the Cytoscape graph."),
                    html.Li("Taxonomy Column: Color-coded by the selected taxonomy category."),
                    html.Li("Numeric Columns: Higher values are represented with deeper colors.")
                ])
            ], className="mb-3"),

            # Treemap Graph Container
            html.H4("Treemap Graph", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Treemap Graph.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "1. ", html.Strong("Hierarchy Representation:"), " Taxa of finer levels (e.g., species) are nested within rectangles of their broader levels (e.g., genus, domain).",
                html.Br(),
                "2. ", html.Strong("Color Coding:"), " Rectangles are color-coded by taxonomic level.",
                html.Ul([
                    html.Li("Darker Colors: Represent finer taxonomic levels, such as species or genus."),
                    html.Li("Lighter Colors: Represent broader taxonomic levels, such as domain or phylum.")
                ]),
                "3. ", html.Strong("Size Representation:"), " The size of each rectangle reflects the total coverage within that taxa.",
                html.Br(),
                "4. ", html.Strong("Click:"), " ",
                html.Ul([
                    html.Li("Click on a rectangle to explore it further in related visualizations."),
                    html.Li("Click on the header of the treemap to return to broader taxonomic levels.")
                ])
            ], className="mb-3"),

            # Cytoscape Graph Container
            html.H4("Cytoscape Graph", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Cytoscape Graph.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "This is a network graph visualizing relationships between annotations or bins based on Hi-C interactions.",
                html.Br(),
                "1. ", html.Strong("Node Distribution:"), " The graph dynamically adjusts positions to emphasize these relationships.",
                html.Ul([
                    html.Li("Nodes are distributed using a force-directed layout. Nodes closer to each other indicate stronger Hi-C interactions."),
                    html.Li("Selected nodes or bins are fixed at the center of the graph for focused analysis."),
                    html.Li("Nodes representing bins are distributed spatially within their annotation groups.")
                ]),
                "2. ", html.Strong("Interactive Node Selection:"), " ",
                html.Ul([
                    html.Li("Click on a node to select it. The selection updates related visualizations, such as the information table, bar chart, and contact table."),
                    html.Li("Selected nodes are visually highlighted with a border, and their connections are emphasized.")
                ])
            ], className="mb-3"),

            # Contact Table Container
            html.H4("Contact Table", className="mt-3"),
            html.Div(
                html.Video(
                    src=r"assets\help\Contact Table.mp4",  # Path to the MP4 file
                    controls=True,  # Enable video controls (play, pause, etc.)
                    width="80%",  # Optional: Set width of the video
                    height="auto",  # Optional: Set height of the video
                ),
                className="d-flex justify-content-center"
            ), 
            html.P([
                "This table displays pairwise Hi-C contact values between taxa, providing a detailed view of their interactions.",
                html.Br(),
                "1. ", html.Strong("Hi-C Contact Values:"), " Each cell represents the interaction strength between the taxa in the corresponding row and column.",
                html.Br(),
                "2. ", html.Strong("Row Annotation Selection:"), " Click on a row to select its corresponding annotation or bin.",
                html.Br(),
                "3. ", html.Strong("Sorting:"), " ",
                html.Ul([
                    html.Li("Click the header of numeric columns to sort rows by the values in ascending or descending order. This helps identify bins or annotations with the strongest or weakest interactions."),
                    html.Li("Click the header of the 'Index' column to reset the sorting and return to the initial state.")
                ]),
                "4. ", html.Strong("Color Coding:"), " ",
                html.Ul([
                    html.Li("Higher contact values are highlighted with deeper colors, making it easy to identify strong interactions at a glance."),
                    html.Li("The row annotation is color-coded consistently with its type, matching the color scheme used in other visualizations.")
                ])
            ], className="mb-3"),
        ],
        style={"maxHeight": "70vh", "overflowY": "auto"}
    )
)

hover_info = {
    'switch-visualization-network':
        (
            "Switch to Normalization Results  \n\n"
            
            "Click this button to switch from the current visualization to the normalization results view."
        ),
        
    'reset-btn': 
        (
            "Reset Selection  \n\n"
            
            "Click this button to clear all selections and reset the visualization to Cross-Taxa Hi-C Interaction."
        ),
        
    "tooltip-toggle-container":
        (
            "Enable or Disable Tooltips  \n\n"
            
            "Check this box to enable tooltips that provide contextual information about the components of this app."
        ),
        
    'dropdowns':
        (
            "Visualization and Selection Dropdowns: Use these dropdown menus to explore different visualization modes or select annotations and bins for detailed analysis.  \n\n"
            
            "The options selected here control the content displayed in the visualizations and tables.  \n\n"
            
            "The dropdown offers three visualization options:  \n"
            "1. **Taxonomic Framework**: Displays a hierarchical treemap showing how annotations are grouped and scaled by a selected metric, such as coverage or classification.  \n"
            "2. **Cross-Taxa Hi-C Interaction**: Focuses on interactions between annotations at a taxonomic level, shown as a Cytoscape graph and bar charts summarizing interaction metrics.  \n"
            "3. **Cross-Bin Hi-C Interactions**: Explores relationships between individual bins and their connections within the dataset, emphasizing specific bins of interest."
        ),
    
    "legand-container":
        (
            "Color Legend and Taxonomy Level Selector:  \n\n"
            
            "1. **Color Legend**: Colors are consistently applied across the Cytoscape graph, bar chart, and tables to represent categories or annotations at the selected taxonomy level. Use the legend to identify categories by their assigned colors.  \n"
            "   - Taxa or bins in the same category share a color system:  \n"
            "       - **Viruses** are reddish colors.  \n"
            "       - **Plasmids** are greenish colors.  \n"
            "       - **Chromosomes** are bluish colors.  \n\n"
            
            "2. **Taxonomy Selector**: The taxonomy selector affects:  \n"
            "   - **Taxonomy Visualization**: Adjusts how nodes are annotated and grouped in the hierarchy, such as by phylum or genus.  \n"
            "   - **Bin Visualization**: Influences how nodes (bins) are distributed in the network, grouping them according to the selected taxonomy level.  \n"
            "   - **Contact Table**: Changes the level of aggregation of the contact table by defining rows and columns based on the selected taxonomy level."
        ),
            
    'bar-chart-container':
        (
            "Bar Chart:  \n\n"
            
            "1. **Chart Types**: The bar chart can display the following types of charts:  \n"
            "   - **Fraction of Classified Bins by Taxonomic Ranks**: Shows the percentage of bins classified at each taxonomic level (e.g., phylum, genus).  \n"
            "   - **Across Taxonomy Hi-C Contacts**: Summarizes Hi-C contact strengths for each taxonomic annotation.  \n"
            "   - **Hi-C Contacts with Selected Bin**: Highlights the contact strengths between the selected bin and other bins.  \n"
            
            "2. **Scroll Bar**: A horizontal scroll bar allows you to navigate through bars when there are too many to display at once.  \n\n"
        ),

    'info-table-container': 
        (
            "Information Table:  \n\n"
            
            "1. **Filter, Sort, and Search**: Use column headers to sort rows or apply filters to narrow down results. You can also use the search box in the headers to find specific bins or annotations quickly.  \n\n"
            
            "2. **Bin Selection**: Click a row to select a bin, updating the Cytoscape graph, bar chart, and other visualizations.  \n\n"
            
            "3. **Automatic Filtering**: The table updates dynamically based on selections.  \n"
            "   - **Taxa Selected**: Shows bins within the selected taxa.  \n"
            "   - **Bin Selected**: Shows bins interacting with the selected bin.  \n\n"

            
            "4. **Filter Checkbox**: Enabling 'Only show elements present in the diagram' checkbox displays only bins or annotations visible in the Cytoscape graph.  \n\n"
            
            "5. **Color Coding**:  \n"
            "   - **Index Column**: Matches the node colors in the Cytoscape graph.  \n"
            "   - **Taxonomy Column**: Color-coded by the selected taxonomy category.  \n"
            "   - **Numeric Columns**: Higher values are represented with deeper colors."
        ),
            
    'treemap-graph-container': 
        (
            "Treemap Graph:  \n\n"
            
            "1. **Hierarchy Representation**: Taxa of finer levels (e.g., species) are nested within rectangles of their broader levels (e.g., genus, domain).  \n\n"
                    
            "2. **Color Coding**: Rectangles are color-coded by taxonomic level.  \n"
            "   - **Darker Colors**: Represent  finer taxonomic levels, such as species or genus.  \n"
            "   - **Lighter Colors**: Represent broader taxonomic levels, such as domain or phylum.  \n\n"
            
            "3. **Size Representation**: The size of each rectangle reflects the total coverage within that taxa.  \n\n"
                        
            "4. **Click**:  \n"
            "   - Click on a rectangle to explore it further in related visualizations.  \n"
            "   - Click on the header of the treemap to return to broader taxonomic levels."
        ),
        
    'cyto-graph-container': 
        (   
            "Cytoscape Graph: This is a network graph visualizing relationships between annotations or bins based on Hi-C interactions.  \n\n"
            
            "1. **Node Distribution**: The graph dynamically adjusts positions to emphasize these relationships.  \n"
            "   - Nodes are distributed using a force-directed layout. Nodes closer to each other indicate stronger Hi-C interactions.  \n"
            "   - Selected nodes or bins are fixed at the center of the graph for focused analysis.  \n"
            "   - Nodes representing bins are distributed spatially within their annotation groups.  \n\n"
            
            "2. **Interactive Node Selection**:  \n"
            "   - Click on a node to select it. The selection updates related visualizations, such as the information table, bar chart, and contact table.  \n"
            "   - Selected nodes are visually highlighted with a border, and their connections are emphasized."
        ),
    
    'contact-table-container': 
        (
            "Contact Table: This table displays pairwise Hi-C contact values between taxa, providing a detailed view of their interactions.  \n\n"
            
            "1. **Hi-C Contact Values**: Each cell represents the interaction strength between the taxa in the corresponding row and column.  \n\n"
            
            "2. **Row Annotation Selection**: Click on a row to select its corresponding annotation or bin.  \n\n"
            
            "3. **Sorting**:  \n"
            "   - Click the header of numeric columns to sort rows by the values in ascending or descending order. This helps identify bins or annotations with the strongest or weakest interactions.  \n"
            "   - Click the header of the 'Index' column to reset the sorting and return to the initial state.  \n\n"
            
            "4. **Color Coding**:  \n"
            "   - Higher contact values are highlighted with deeper colors, making it easy to identify strong interactions at a glance.  \n"
            "   - The row annotation is color-coded consistently with its type, matching the color scheme used in other visualizations."
        )
}