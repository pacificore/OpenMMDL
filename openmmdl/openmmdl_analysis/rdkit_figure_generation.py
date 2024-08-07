import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import cairosvg
import pylab
import os
import MDAnalysis as mda


class LigandImageGenerator:
    def __init__(
        self,
        ligand_name,
        complex_pdb_file,
        ligand_no_h_pdb_file,
        smiles_file,
        output_svg_filename,
        fig_type="svg",
    ):
        """
        Initialize the LigandImageGenerator class.

        Args:
            ligand_name (str): Name of the ligand in the protein-ligand complex topology.
            complex_pdb_file (str): Path to the protein-ligand complex PDB file.
            ligand_no_h_pdb_file (str): Path to the ligand PDB file without hydrogens.
            smiles_file (str): Path to the SMILES file with the reference ligand.
            output_svg_filename (str): Name of the output SVG file.
            fig_type (str): Type of the output figure. Can be "svg" or "png".
        """
        self.ligand_name = ligand_name
        self.complex_pdb_file = complex_pdb_file
        self.ligand_no_h_pdb_file = ligand_no_h_pdb_file
        self.smiles_file = smiles_file
        self.output_svg_filename = output_svg_filename
        self.fig_type = fig_type

    def generate_image(self):
        """Generates an SVG image (or PNG) of the ligand."""
        try:
            # Load complex and ligand structures
            complex = mda.Universe(self.complex_pdb_file)
            ligand_no_h = mda.Universe(self.ligand_no_h_pdb_file)
            lig_noh = ligand_no_h.select_atoms("all")
            complex_lig = complex.select_atoms(f"resname {self.ligand_name}")

            # Load ligand from PDB file
            mol = Chem.MolFromPDBFile(self.ligand_no_h_pdb_file)
            lig_rd = mol

            # Load reference SMILES
            with open(self.smiles_file, "r") as file:
                reference_smiles = file.read().strip()
            reference_mol = Chem.MolFromSmiles(reference_smiles)

            # Prepare ligand
            prepared_ligand = AllChem.AssignBondOrdersFromTemplate(
                reference_mol, lig_rd
            )
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

            # Generate an SVG image of the ligand
            drawer = Draw.MolDraw2DSVG(5120, 3200)
            drawer.drawOptions().addStereoAnnotation = (
                True  # Add stereo information if available
            )
            drawer.DrawMolecule(prepared_ligand)

            # Adjust font size in the SVG output using the FontSize method
            font_size = drawer.FontSize()
            drawer.SetFontSize(
                font_size * 0.5
            )  # You can adjust the multiplier as needed

            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace("svg:", "")

            # Save the SVG image to the specified output file
            with open(self.output_svg_filename, "w") as f:
                f.write(svg)

            # Convert to PNG if requested
            if self.fig_type == "png":
                png_filename = self.output_svg_filename.replace(".svg", ".png")
                cairosvg.svg2png(url=self.output_svg_filename, write_to=png_filename)
                print(f"PNG image saved as: {png_filename}")

        except Exception as e:
            print(f"Error: {e}")


class InteractionProcessor:
    def __init__(self, complex_pdb_file, ligand_no_h_pdb_file):
        """
        Initialize the InteractionProcessor class.

        Args:
            complex_pdb_file (str): Path to the protein-ligand complex PDB file.
            ligand_no_h_pdb_file (str): Path to the ligand PDB file without hydrogens.
        """
        self.complex_pdb_file = complex_pdb_file
        self.ligand_no_h_pdb_file = ligand_no_h_pdb_file
        self.complex = mda.Universe(complex_pdb_file)
        self.ligand_no_h = mda.Universe(ligand_no_h_pdb_file)
        self.lig_noh = self.ligand_no_h.select_atoms("all")

    def split_interaction_data(self, data):
        """Splits the input data into multiple parts.

        Args:
            data (list): A list of ResNr and ResType, Atom indices, interaction type that needs to be split.

        Returns:
            list: A new list of the interaction data that consists of three parts.
        """
        split_data = []
        for item in data:
            parts = item.split("_")
            protein_partner_name = parts[0]
            numeric_codes = " ".join(parts[1:-1])
            interaction_type = parts[-1]
            split_value = f"{protein_partner_name} {numeric_codes} {interaction_type}"
            split_data.append(split_value)
        return split_data

    def highlight_numbers(self, split_data, starting_idx):
        """Extracts the data from the split_data output of the interactions and categorizes it to its respective list.

        Args:
            split_data (list): A list of interaction data items, where each item contains information about protein partner name,
            numeric codes and interaction type.
            starting_idx (list): Starting index of the ligand atom indices used for identifying the correct atom to highlight.

        Returns:
            tuple: A tuple that contains list of all of the highlighted atoms of all of the interactions.
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
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    if type == "Donor":
                        highlighted_hbond_donor.append(lig_real_index - 1)
                    elif type == "Acceptor":
                        highlighted_hbond_acceptor.append(lig_real_index - 1)

            elif interaction_type == "hydrophobic":
                for code in numeric_codes:
                    atom_index = int(code)
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_hydrophobic.append(lig_real_index - 1)

            elif interaction_type == "waterbridge":
                for code in numeric_codes:
                    atom_index = int(code)
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_waterbridge.append(lig_real_index - 1)

            elif interaction_type == "pistacking":
                split_codes = numeric_codes[0].split(",")
                for code in split_codes:
                    atom_index = int(code)
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_pistacking.append(lig_real_index - 1)

            elif interaction_type == "halogen":
                numeric_codes = parts[1:-2]
                for code in numeric_codes:
                    atom_index = int(code)
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_halogen.append(lig_real_index - 1)

            elif interaction_type == "saltbridge":
                numeric_codes = parts[1:-3]
                saltbridge_type = parts[-2]
                if saltbridge_type == "NI":
                    split_codes = numeric_codes[0].split(",")
                    for code in split_codes:
                        atom_index = int(code)
                        complex_id = self.complex.select_atoms(f"id {atom_index}")
                        for atom in complex_id:
                            atom_name = atom.name
                        for lig_atom in self.lig_noh:
                            if lig_atom.name == atom_name:
                                lig_real_index = lig_atom.id
                        highlighted_ni.append(lig_real_index - 1)
                elif saltbridge_type == "PI":
                    for code in numeric_codes:
                        atom_index = int(code)
                        complex_id = self.complex.select_atoms(f"id {atom_index}")
                        for atom in complex_id:
                            atom_name = atom.name
                        for lig_atom in self.lig_noh:
                            if lig_atom.name == atom_name:
                                lig_real_index = lig_atom.id
                        highlighted_pi.append(lig_real_index - 1)

            elif interaction_type == "pication":
                numeric_codes = parts[1:-2]
                for code in numeric_codes:
                    atom_index = int(code)
                    complex_id = self.complex.select_atoms(f"id {atom_index}")
                    for atom in complex_id:
                        atom_name = atom.name
                    for lig_atom in self.lig_noh:
                        if lig_atom.name == atom_name:
                            lig_real_index = lig_atom.id
                    highlighted_pication.append(lig_real_index - 1)

            elif interaction_type == "metal":
                ligidx = parts[1]
                atom_index = int(ligidx)
                complex_id = self.complex.select_atoms(f"id {atom_index}")
                for atom in complex_id:
                    atom_name = atom.name
                for lig_atom in self.lig_noh:
                    if lig_atom.name == atom_name:
                        lig_real_index = lig_atom.id
                highlighted_metal.append(lig_real_index - 1)

        for value in highlighted_hbond_donor[:]:
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

    def generate_interaction_dict(self, interaction_type, keys):
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

        interaction_dict = {
            int(key): interaction_dict[interaction_type] for key in keys
        }
        return interaction_dict

    def update_dict(self, target_dict, *source_dicts):
        """Updates the dictionary with the keys and values from other dictionaries.

        Args:
            target_dict (dict): The dictionary that needs to be updated with new keys and values.
            source_dicts (dict): One or multiple dictionaries that are used to update the target dictionary with new keys and values.
        """
        for source_dict in source_dicts:
            for key, value in source_dict.items():
                int_key = int(key)
                if int_key not in target_dict:
                    target_dict[int_key] = value


class ImageMerger:
    def __init__(
        self, binding_mode, occurrence_percent, split_data, merged_image_paths
    ):
        self.binding_mode = binding_mode
        self.occurrence_percent = occurrence_percent
        self.split_data = split_data
        self.merged_image_paths = merged_image_paths

    def create_and_merge_images(self):
        """Create and merge images to generate a legend for binding modes.

        Returns:
            list: Paths to the merged images.
        """
        # Create the main figure and axis
        fig = pylab.figure()
        ax = fig.add_subplot(111)

        # Data for the x-axis and random data for demonstration
        x = range(10)
        data_points = [pylab.randn(10) for _ in range(len(self.split_data))]

        # Plot lines on the same axis and collect them into a list
        lines = []
        filtered_split_data = [
            entry for entry in self.split_data if "FRAME" not in entry
        ]
        for i, data in enumerate(filtered_split_data):
            y = data_points[i]
            label = data.split()[-1]
            type = data.split()[-2]
            if label == "hydrophobic":
                (line,) = ax.plot(
                    x, y, label=data, color=(1.0, 1.0, 0.0), linewidth=5.0
                )
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
                (line,) = ax.plot(
                    x, y, label=data, color=(1.0, 0.0, 0.9), linewidth=5.0
                )
            elif label == "pistacking":
                (line,) = ax.plot(
                    x, y, label=data, color=(0.0, 0.0, 1.0), linewidth=5.0
                )
            elif label == "pication":
                (line,) = ax.plot(
                    x, y, label=data, color=(0.0, 0.0, 1.0), linewidth=5.0
                )
            elif label == "waterbridge":
                (line,) = ax.plot(
                    x, y, label=data, color=(0.0, 1.0, 0.9), linewidth=5.0
                )
            elif label == "metal":
                (line,) = ax.plot(
                    x, y, label=data, color=(1.0, 0.6, 0.0), linewidth=5.0
                )
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
        figlegend.text(
            0.5, 0.9, f"{self.binding_mode}", ha="center", fontsize=12, weight="bold"
        )
        figlegend.text(
            0.5,
            0.85,
            f"Occurrence {self.occurrence_percent}%",
            ha="center",
            fontsize=12,
            weight="bold",
        )

        # Save the legend figure to a file
        legend_filename = f"{self.binding_mode}_legend.png"
        figlegend.savefig(legend_filename)

        # Read the two images
        image1 = Image.open(f"{self.binding_mode}.png")
        image2 = Image.open(legend_filename)

        # Resize the first image
        image1_size = image1.size
        image2_size = image2.size
        total_width = image1_size[0] + image2_size[0]
        new_image = Image.new("RGB", (total_width, image1_size[1]))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_size[0], 0))

        # Save the merged image
        merged_image_filename = f"{self.binding_mode}_merged.png"
        new_image.save(merged_image_filename, "PNG")

        # Append the merged image path to the list
        self.merged_image_paths.append(merged_image_filename)

        # Remove the original files
        os.remove(f"{self.binding_mode}.png")
        os.remove(legend_filename)
        os.remove(f"{self.binding_mode}.svg")

        return self.merged_image_paths


class FigureArranger:
    def __init__(self, merged_image_paths, output_path):
        self.merged_image_paths = merged_image_paths
        self.output_path = output_path

    def arranged_figure_generation(self):
        """Generate an arranged figure by arranging merged images in rows and columns."""
        # Open the list of images
        merged_images = [Image.open(path) for path in self.merged_image_paths]

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
        big_figure.save(self.output_path, "PNG")

        # Rename the merged image
        os.rename(
            self.output_path,
            "Binding_Modes_Markov_States/" + os.path.basename(self.output_path),
        )

        # Remove the individual image files
        for path in self.merged_image_paths:
            os.remove(path)
