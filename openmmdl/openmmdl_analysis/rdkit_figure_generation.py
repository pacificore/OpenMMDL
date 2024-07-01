import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import cairosvg
import pylab
import os
import MDAnalysis as mda


def generate_ligand_image(
    ligand_name,
    complex_pdb_file,
    ligand_no_h_pdb_file,
    output_svg_filename,
):
    """Generates a SVG image of the ligand.

    Args:
        ligand_name (str): Name of the ligand in the protein-ligand complex topology.
        complex_pdb_file (str): Path to the protein-ligand complex PDB file.
        ligand_no_h_pdb_file (str): Path to the ligand PDB file without hydrogens.
        output_svg_filename (str): Name of the output SVG file.
    """
    try:
        # Load complex and ligand structures
        complex = mda.Universe(complex_pdb_file)
        complex_lig = complex.select_atoms(f"resname {ligand_name}")
        # Load ligand without hydrogens for checking correctness atom number
        ligand_no_h = mda.Universe(ligand_no_h_pdb_file)
        lig_noh = ligand_no_h.select_atoms("all")
        # Application of RDKit Converter to obtain rdkit mol of ligand
        lig_atoms = complex_lig.convert_to("RDKIT")
        prepared_ligand = Chem.RemoveAllHs(lig_atoms)
        AllChem.Compute2DCoords(prepared_ligand)

        # Map atom indices between ligand_no_h and complex
        for atom in prepared_ligand.GetAtoms():
            atom_index = atom.GetIdx()
            for lig_atom in lig_noh:
                lig_index = lig_atom.index
                if atom_index == lig_index:
                    lig_atom_name = lig_atom.name
                    for comp_lig in complex_lig:
                        comp_lig_name = comp_lig.name
                        if lig_atom_name == comp_lig_name:
                            num = int(comp_lig.id)
                            atom.SetAtomMapNum(num)

        # Generate a SVG image of the ligand without highlighting atoms
        drawer = Draw.MolDraw2DSVG(5120, 3200)
        drawer.drawOptions().addStereoAnnotation = (
            True  # Add stereo information if available
        )
        drawer.DrawMolecule(prepared_ligand)

        # Adjust font size in the SVG output using the FontSize method
        font_size = drawer.FontSize()
        drawer.SetFontSize(font_size * 0.5)  # You can adjust the multiplier as needed

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")

        # Save the SVG image to the specified output file
        with open(output_svg_filename, "w") as f:
            f.write(svg)

    except Exception as e:
        print(f"Error: {e}")


def split_interaction_data(data):
    """Splits the Input which consists of the ResNr and ResType, Atom indices, interaction type in multiple parts.

    Args:
        data (list): A list of ResNr and ResType, Atom indices, interaction type that needs to be split.

    Returns:
        list: A new list of the interaction data that consists of three parts, being the protein_partner_name that represents the interacting protein residue, numeric codes, that represent the atom indices of the interacting atoms of the ligand and the interaction type.
    """
    split_data = []
    for item in data:
        parts = item.split("_")
        protein_partner_name = parts[0]
        numeric_codes = " ".join(parts[1:-1])  # Join numeric codes with spaces
        interaction_type = parts[-1]
        split_value = f"{protein_partner_name} {numeric_codes} {interaction_type}"
        split_data.append(split_value)

    return split_data


def highlight_numbers(split_data, starting_idx):
    """Extracts the data from the split_data output of the interactions and categorizes it to its respective list.

    Args:
        split_data (list): A list of interaction data items, where each item contains information about protein partner name, numeric codes and interaction type.
        starting_idx (list): Starting index of the ligand atom indices used for identifying the correct atom to highlight.

    Returns:
        tuple: A tuple that contains list of all of the highlighted atoms of all of the interactions.
        - highlighted_hbond_donor (list of int): Atom indices for hydrogen bond donors.
        - highlighted_hbond_acceptor (list of int): Atom indices for hydrogen bond acceptors.
        - highlighted_hbond_both (list of int): Atom indices for interactions that are both donors and acceptors.
        - highlighted_hydrophobic (list of int): Atom indices for hydrophobic interactions.
        - highlighted_waterbridge (list of int): Atom indices for water-bridge interactions.
        - highlighted_pistacking (list of int): Atom indices for pi-stacking interactions.
        - highlighted_halogen (list of int): Atom indices for halogen interactions.
        - highlighted_ni (list of int): Atom indices for negative ionizable salt bridge interactions.
        - highlighted_pi (list of int): Atom indices for positive ionizable salt bridge interactions.
        - highlighted_pication (list of int): Atom indices for pi-cation interactions.
        - highlighted_metal (list of int): Atom indices for metal interactions.
    """
    highlighted_hbond_acceptor = []
    highlighted_hbond_donor = []
    highlighted_hydrophobic = []
    highlighted_hbond_both = []
    highlighted_waterbridge = []
    highlighted_pistacking = []
    highlighted_halogen = []
    highlighted_ni = []
    highlighted_pi = []
    highlighted_pication = []
    highlighted_metal = []
    complex = mda.Universe("complex.pdb")
    ligand_no_h = mda.Universe("lig_no_h.pdb")
    lig_noh = ligand_no_h.select_atoms("all")

    for item in split_data:
        parts = item.split()
        protein_partner_name = parts[0]
        numeric_codes = parts[1:-1]
        interaction_type = parts[-1]

        if interaction_type == "hbond":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-2]
            type = parts[-2]
            interaction_type = parts[-1]
            for code in numeric_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                if type == "Donor":
                    highlighted_hbond_donor.append(lig_real_index - 1)
                elif type == "Acceptor":
                    highlighted_hbond_acceptor.append(lig_real_index - 1)

        elif interaction_type == "hydrophobic":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-1]
            interaction_type = parts[-1]
            for code in numeric_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_hydrophobic.append(lig_real_index - 1)

        elif interaction_type == "waterbridge":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-2]
            interaction_type = parts[-1]
            for code in numeric_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_waterbridge.append(lig_real_index - 1)

        elif interaction_type == "pistacking":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-1]
            print(numeric_codes)
            interaction_type = parts[-1]
            split_codes = numeric_codes[0].split(",")
            print(split_codes)
            for code in split_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_pistacking.append(lig_real_index - 1)

        elif interaction_type == "halogen":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-2]
            interaction_type = parts[-1]
            halogen_type = parts[-2]
            for code in numeric_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_halogen.append(lig_real_index - 1)

        elif interaction_type == "saltbridge":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-3]
            interaction_type = parts[-1]
            saltbridge_type = parts[-2]
            if saltbridge_type == "NI":
                split_codes = numeric_codes[0].split(",")
                numeric_values = [int(code) for code in split_codes]
                for code in numeric_values:
                    atom_index = int(code)
                    complex_id = complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_ni.append(lig_real_index - 1)
            if saltbridge_type == "PI":
                for code in numeric_codes:
                    atom_index = int(code)
                    complex_id = complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_pi.append(lig_real_index - 1)

        elif interaction_type == "pication":
            parts = item.split()
            protein_partner_name = parts[0]
            numeric_codes = parts[1:-2]
            interaction_type = parts[-1]
            halogen_type = parts[-2]
            for code in numeric_codes:
                atom_index = int(code)
                complex_id = complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_pication.append(lig_real_index - 1)

        elif interaction_type == "metal":
            parts = item.split()
            special_ligand = parts[0]
            ligidx = parts[1]
            metal_type = parts[2]
            atom_index = int(ligidx)
            complex_id = complex.select_atoms(f"id {atom_index}")
            for atom in complex_id:
                atom_name = atom.name
            for lig_atom in lig_noh:
                if lig_atom.name == atom_name:
                    lig_real_index = lig_atom.id
            highlighted_metal.append(lig_real_index - 1)

    for value in highlighted_hbond_donor[
        :
    ]:  # Using a copy of the list to avoid modifying while iterating
        if value in highlighted_hbond_acceptor:
            highlighted_hbond_donor.remove(value)
            highlighted_hbond_acceptor.remove(value)
            highlighted_hbond_both.append(value)

    return (
        highlighted_hbond_donor,
        highlighted_hbond_acceptor,
        highlighted_hbond_both,
        highlighted_hydrophobic,
        highlighted_waterbridge,
        highlighted_pistacking,
        highlighted_halogen,
        highlighted_ni,
        highlighted_pi,
        highlighted_pication,
        highlighted_metal,
    )


def generate_interaction_dict(interaction_type, keys):
    """Generates a dictionary of interaction RGB color model based on the provided interaction type.

    Args:
        interaction_type (str): The type of the interaction, for example 'hydrophobic'.
        keys (list): List of the highlighted atoms that display an interaction.

    Returns:
        dict: A dictionary with the interaction types are associated with their respective RGB color codes.
    """
    interaction_dict = {
        "hbond_acceptor": (1.0, 0.6, 0.6),
        "hbond_both": (0.6, 0.0, 0.5),
        "hbond_donor": (0.3, 0.5, 1.0),
        "hydrophobic": (1.0, 1.0, 0.0),
        "waterbridge": (0.0, 1.0, 0.9),
        "pistacking": (0.0, 0.0, 1.0),
        "halogen": (1.0, 0.0, 0.9),
        "ni": (0.0, 0.0, 1.0),
        "pi": (1.0, 0.0, 0.0),
        "pication": (0.0, 0.0, 1.0),
        "metal": (1.0, 0.6, 0.0),
    }

    interaction_dict = {int(key): interaction_dict[interaction_type] for key in keys}

    return interaction_dict


def update_dict(target_dict, *source_dicts):
    """Updates the dictionary wth the keys and values from other dictionaries.

    Args:
        target_dict (dict): The dictionary that needs to be updated with new keys and values.
        source_dicts (dict): One or multiple dictionaries that are used to update the target dictionary with new keys and values.
    """
    for source_dict in source_dicts:
        for key, value in source_dict.items():
            int_key = int(key)  # Convert the key to an integer
            if int_key not in target_dict:
                target_dict[int_key] = value


def create_and_merge_images(
    binding_mode, occurrence_percent, split_data, merged_image_paths
):
    """Create and merge images to generate a legend for binding modes.

    Args:
        binding_mode (str): Name of the binding mode.
        occurrence_percent (float): The percentage of the binding mode occurrence.
        split_data (list): Data of the interactions used to generate the legend.
        merged_image_paths (list): A list with the paths to the rdkit figures.

    Returns:
        list: Paths to the merged images.
    """
    # Create the main figure and axis
    fig = pylab.figure()
    ax = fig.add_subplot(111)

    # Data for the x-axis and random data for demonstration
    x = range(10)
    data_points = [pylab.randn(10) for _ in range(len(split_data))]

    # Plot lines on the same axis and collect them into a list
    lines = []
    filtered_split_data = [entry for entry in split_data if "FRAME" not in entry]
    for i, data in enumerate(filtered_split_data):
        y = data_points[i]
        label = data.split()[-1]
        type = data.split()[-2]
        if label == "hydrophobic":
            (line,) = ax.plot(x, y, label=data, color=(1.0, 1.0, 0.0), linewidth=5.0)
        elif label == "hbond":
            if type == "Acceptor":
                (line,) = ax.plot(
                    x, y, label=data, color=(1.0, 0.6, 0.6), linewidth=5.0
                )
            elif type == "Donor":
                (line,) = ax.plot(
                    x, y, label=data, color=(0.3, 0.5, 1.0), linewidth=5.0
                )
        elif label == "halogen":
            (line,) = ax.plot(x, y, label=data, color=(1.0, 0.0, 0.9), linewidth=5.0)
        elif label == "pistacking":
            (line,) = ax.plot(x, y, label=data, color=(0.0, 0.0, 1.0), linewidth=5.0)
        elif label == "pication":
            (line,) = ax.plot(x, y, label=data, color=(0.0, 0.0, 1.0), linewidth=5.0)
        elif label == "waterbridge":
            (line,) = ax.plot(x, y, label=data, color=(0.0, 1.0, 0.9), linewidth=5.0)
        elif label == "metal":
            (line,) = ax.plot(x, y, label=data, color=(1.0, 0.6, 0.0), linewidth=5.0)
        elif label == "saltbridge":
            if type == "NI":
                (line,) = ax.plot(
                    x, y, label=data, color=(0.0, 0.0, 1.0), linewidth=5.0
                )
            elif type == "PI":
                (line,) = ax.plot(
                    x, y, label=data, color=(1.0, 0.0, 0.0), linewidth=5.0
                )
        else:
            (line,) = ax.plot(x, y, label=data)
        lines.append(line)

    # Create a separate figure for the legend
    figlegend = pylab.figure(figsize=(8, 6))

    # Add a legend to the subplot (ax) using the lines and full entries as labels
    legend = figlegend.legend(lines, filtered_split_data, loc="center")

    # Set the linewidth of the legend lines to be thicker
    for line in legend.get_lines():
        line.set_linewidth(5.0)

    # Add text above the legend
    figlegend.text(0.5, 0.9, f"{binding_mode}", ha="center", fontsize=12, weight="bold")
    figlegend.text(
        0.5,
        0.85,
        f"Occurrence {occurrence_percent}%",
        ha="center",
        fontsize=12,
        weight="bold",
    )

    # Save the legend figure to a file
    legend_filename = f"{binding_mode}_legend.png"
    figlegend.savefig(legend_filename)

    # Read the two images
    image1 = Image.open(f"{binding_mode}.png")
    image2 = Image.open(legend_filename)

    # Resize the first image
    image1_size = image1.size
    image2_size = image2.size
    total_width = image1_size[0] + image2_size[0]
    new_image = Image.new("RGB", (total_width, image1_size[1]))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))

    # Save the merged image
    merged_image_filename = f"{binding_mode}_merged.png"
    new_image.save(merged_image_filename, "PNG")

    # Append the merged image path to the list
    merged_image_paths.append(merged_image_filename)

    # Remove the original files
    os.remove(f"{binding_mode}.png")
    os.remove(legend_filename)
    os.remove(f"{binding_mode}.svg")

    return merged_image_paths


def arranged_figure_generation(merged_image_paths, output_path):
    """Generate an arranged figure by arranging merged images in rows and columns.

    Args:
        merged_image_paths (list): Paths of the merged images with the rdkit figure and legend.
        output_path (dict): The paths where the arranged output should be saved.
    """
    # Open the list of images
    merged_images = [Image.open(path) for path in merged_image_paths]

    # Calculate the maximum width and height for the images
    max_width = max(image.size[0] for image in merged_images)
    max_height = max(image.size[1] for image in merged_images)

    # Determine the number of images per row (in your case, 2 images per row)
    images_per_row = 2

    # Calculate the number of rows and columns required
    num_rows = (len(merged_images) + images_per_row - 1) // images_per_row
    total_width = max_width * images_per_row
    total_height = max_height * num_rows

    # Create a new image with the calculated width and height
    big_figure = Image.new(
        "RGB", (total_width, total_height), (255, 255, 255)
    )  # Set background to white

    x_offset = 0
    y_offset = 0

    for image in merged_images:
        # Paste the image onto the big figure
        big_figure.paste(image, (x_offset, y_offset))

        # Update offsets
        x_offset += max_width

        # Move to the next row if necessary
        if x_offset >= total_width:
            x_offset = 0
            y_offset += max_height

    # Save the big figure
    big_figure.save(output_path, "PNG")

    # Rename the merged image
    os.rename(
        output_path, "Binding_Modes_Markov_States/" + os.path.basename(output_path)
    )

    # Remove the individual image files
    for path in merged_image_paths:
        os.remove(path)
