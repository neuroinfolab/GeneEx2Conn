# GeneEx2Conn/harmonize/data_utils.py

from imports import *

# useful global paths
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = par_dir + '/GeneEx2Conn_data'
samples_dir = data_dir + '/samples_files'

def find_gene_intersection(data_dir):
    # Initialize a set to store the intersection of genes
    gene_intersection = None

    # Iterate over each txt file in the directory
    for txt_file in os.listdir(data_dir):
        # Check if the file is a txt file
        if txt_file.endswith('.txt'):
            # Read the gene list from the file
            with open(os.path.join(data_dir, txt_file), 'r') as file:
                gene_set = set(file.read().splitlines())

                # If it's the first file, start the intersection with this set
                if gene_intersection is None:
                    gene_intersection = gene_set
                else:
                    # Intersect with the current set of genes
                    gene_intersection &= gene_set

    # If there are no intersecting genes, handle it appropriately
    if gene_intersection is None:
        return set(), 0

    # Return the intersecting genes and their count
    return gene_intersection, len(gene_intersection)


def clean_csv_metadata(csv_paths, intersecting_genes): 
    """
    Function to categorize sex, age, ethnicity if possible,
    retain only overlapping genes, and create an age decade column.
    """
    sample_dfs = []

    for csv_path in csv_paths:
        dataset_df = pd.read_csv(csv_path)

        # Keep only the first 2 characters in the age column (assumed to be the numeric part)
        dataset_df['age'] = dataset_df['age'].astype(str).str[:2].astype(int)  # Convert to int after slicing

        # Create the age decade column
        def assign_age_decade(age):
            if age >= 10:
                return f'{age // 10 * 10}-{age // 10 * 10 + 9}'
            else:
                return '0-9'  # Handle very young ages, if any
        
        dataset_df['age decade'] = dataset_df['age'].apply(assign_age_decade)
        
        # Move the 'age_decade' column right after 'age'
        age_idx = dataset_df.columns.get_loc('age')
        cols = dataset_df.columns.tolist()
        cols.insert(age_idx + 1, cols.pop(cols.index('age decade')))
        dataset_df = dataset_df[cols]  # Reorder the columns
        # Map sex: make M=1, F=2
        dataset_df['sex'] = dataset_df['sex'].replace({1: 'M', 2: 'F'})

        # Placeholder for ethnicity processing (you can add more entries to ethnicity_dict later)
        ethnicities = ['Caucasian', 'Hispanic', 'African', 'African American', 'Asian', 'Caribbean', 'Middle Eastern']
        ethnicity_dict = {'Caucasian': ['Caucasian', 'European']}

        # Dynamically find where metadata ends and gene columns start
        metadata_end_column = 'sequencing_type'
        metadata_columns = dataset_df.columns[:dataset_df.columns.get_loc(metadata_end_column) + 1].tolist()

        # Find the intersecting genes that are present in the DataFrame and sort them alphabetically
        gene_columns = [col for col in dataset_df.columns if col not in metadata_columns and col in intersecting_genes]
        sorted_gene_columns = sorted(gene_columns)

        # Combine metadata columns with the sorted gene columns
        cols_to_keep = metadata_columns + sorted_gene_columns

        # Filter the DataFrame to keep only the necessary columns
        dataset_df = dataset_df[cols_to_keep]
        
        sample_dfs.append(dataset_df)
    
    return sample_dfs


# Load the MNI nifti file and extract data
def load_mni_atlas(atlas_path):
    """
    Load the MNI atlas using nibabel and return the data and affine matrix.
    """
    mni_img = nib.load(atlas_path)
    mni_data = mni_img.get_fdata()
    affine = mni_img.affine
    return mni_data, affine


# Sample usage
mni_nifti_path = data_dir+'/atlas_info/MNI-maxprob-thr0-1mm.nii'
mni_xml_path = '/path_to_mni_atlas/MNI.xml'

# Load the MNI atlas and the label mapping (you'll need to parse the MNI.xml to get this mapping)
mni_data, affine = load_mni_atlas(mni_nifti_path)

# List of regions to be labeled as 'Subcortical Microstructure'
subcortical_microstructure_regions = [
    'LH-GPe', 'LH-GPi', 'LH-VeP', 'LH-HTH', 'LH-SNc_PBP_VTA', 'LH-RN', 'LH-SNr',
    'RH-GPe', 'RH-GPi', 'RH-RN', 'RH-SNr', 'RH-HTH', 'RH-MN', 'RH-SNc_PBP_VTA',
    'Hypothalamus', 'Substantia nigra', 'Globus Pallidus', 'Internal Capsule', 'Substantia Nigra'
]

# Add the MNI region mapping based on the MNI.xml file.
mni_label_map = {
    1:'Caudate',
    2:'Cerebellum',
    3:'Frontal Lobe',
    4:'Insula',
    5:'Occipital Lobe',
    6:'Parietal Lobe',
    7:'Putamen',
    8:'Temporal Lobe',
    9:'Thalamus'
}

# Function to map MNI region from coordinates
def get_mni_region_from_coords(coords):
    """
    Given a list of coordinates, determine the most frequent MNI macro region.
    If no match is found, use the closest XML centroid.
    
    Args:
        coords (list of tuples): List of 3D MNI coordinates.
        mni_data (numpy array): The loaded MNI atlas data.
        affine (numpy array): Affine transformation matrix from the NIfTI file.
        mni_label_map (dict): Dictionary mapping MNI ROI values to labels.

    Returns:
        str: Most frequent MNI region label.
    """
    regions = []
    
    for coord in coords:
        if coord is None:
            continue  # Skip if the coordinate is None

        # Transform MNI coordinates to voxel indices using the affine
        voxel_coord = nib.affines.apply_affine(np.linalg.inv(affine), coord)
        voxel_coord = np.round(voxel_coord).astype(int)

        # Ensure the voxel coordinates are within the MNI data bounds
        if (0 <= voxel_coord[0] < mni_data.shape[0] and
            0 <= voxel_coord[1] < mni_data.shape[1] and
            0 <= voxel_coord[2] < mni_data.shape[2]):
            region_value = mni_data[tuple(voxel_coord)]
            region_label = mni_label_map.get(int(region_value), None)
            
            # If a region label is found in the MNI data, add it
            if region_label:
                regions.append(region_label)
            else:
                continue  # Skip unknown values in MNI data
        else:
            continue  # Skip out-of-bounds coordinates

    if regions:
        most_common_region = Counter(regions).most_common(1)[0][0]
        return most_common_region
    else:
        return 'Unknown'


# Function to assign macro regions based on centroid coordinates
def assign_macro_regions(combined_df):
    """
    Assign a macro brain region to each sample based on centroid coordinates.
    
    Args:
        combined_df (pd.DataFrame): DataFrame containing the centroid coordinates.
        mni_data (numpy array): The loaded MNI atlas data.
        affine (numpy array): Affine transformation matrix from the NIfTI file.
        mni_label_map (dict): Dictionary mapping MNI ROI values to labels.
        
    Returns:
        pd.DataFrame: Updated DataFrame with the macro region column added.
    """
    # List of regions to be labeled as 'Subcortical Microstructure'
    subcortical_microstructure_regions = [
        'LH-GPe', 'LH-GPi', 'LH-VeP', 'LH-HTH', 'LH-SNc_PBP_VTA', 'LH-RN', 'LH-SNr',
        'RH-GPe', 'RH-GPi', 'RH-RN', 'RH-SNr', 'RH-HTH', 'RH-MN', 'RH-SNc_PBP_VTA',
        'Hypothalamus', 'Substantia nigra', 'Globus Pallidus', 'Internal Capsule', 'Substantia Nigra'
    ]
    
    # Create a new column to store the macro regions
    macro_regions = []

    for idx, row in combined_df.iterrows():
        centroid_coords = row['centroid coordinates']
        tissue_local_name = row['tissue sample local name']

        if centroid_coords is not None:
            if isinstance(centroid_coords, str):
                centroid_coords = eval(centroid_coords)
            
            if isinstance(centroid_coords, list):
                valid_coords = [coord for coord in centroid_coords if coord is not None]
                
                if valid_coords:
                    macro_region = get_mni_region_from_coords(valid_coords)
                    if macro_region == 'Unknown' and tissue_local_name in subcortical_microstructure_regions:
                        macro_region = 'Subcortical Microstructure'
                else:
                    macro_region = 'Unknown'
            else:
                macro_region = 'Unknown'
        else:
            # Check if the local tissue name falls under the subcortical microstructure regions
            if tissue_local_name in subcortical_microstructure_regions:
                macro_region = 'Subcortical Microstructure'
            else:
                macro_region = 'Unknown'
        
        macro_regions.append(macro_region)

    # Insert the macro region column into the DataFrame after the centroid coordinates column
    combined_df.insert(combined_df.columns.get_loc('centroid coordinates') + 1, 'macro region', macro_regions)
    
    return combined_df