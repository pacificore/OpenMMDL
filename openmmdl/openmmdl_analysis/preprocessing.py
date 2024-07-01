import MDAnalysis as mda
import subprocess
import os
import re
import rdkit
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, SDWriter
from rdkit.Chem.Draw import rdMolDraw2D


def renumber_protein_residues(input_pdb, reference_pdb, output_pdb):
    """Renumber protein residues in a molecular dynamics trajectory based on a reference structure.

    Args:
        input_pdb (str): Path to the input PDB file representing the molecular dynamics trajectory to be renumbered.
        reference_pdb (str): Path to the reference PDB file representing the molecular dynamics trajectory used as a reference.
        output_pdb (str): Path to the output PDB file where the renumbered trajectory will be saved..
    """
    # Load trajectories
    traj_input = md.load(input_pdb)
    traj_reference = md.load(reference_pdb)

    # Get the topology DataFrames
    input_top_df, _ = traj_input.topology.to_dataframe()
    ref_top_df, _ = traj_reference.topology.to_dataframe()

    # List of common protein residue names including additional residues
    protein_residues = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
        "HIE",
        "HID",
        "HIP",
    ]

    # Iterate over each chain in the reference topology
    for chain_id in ref_top_df["chainID"].unique():
        # Extract residue indices from the reference topology for the current chain
        ref_residue_indices = (
            ref_top_df[
                (ref_top_df["chainID"] == chain_id)
                & ref_top_df["resName"].isin(protein_residues)
            ]["resSeq"].values
            - 1
        )

        # Update residue indices in the input topology DataFrame for the current chain
        mask = (input_top_df["chainID"] == chain_id) & input_top_df["resName"].isin(
            protein_residues
        )
        input_top_df.loc[mask, "resSeq"] = ref_residue_indices + 1

    # Create a new topology from the modified DataFrame
    new_top = md.Topology.from_dataframe(input_top_df)

    # Update the topology in the input trajectory
    new_traj = traj_input.slice(range(traj_input.n_frames))
    new_traj.topology = new_top

    # Save the renumbered trajectory to a new PDB file
    new_traj.save(output_pdb)


def increase_ring_indices(ring, lig_index):
    """Increases the atom indices in a ring of the ligand obtained from the ligand to fit the atom indices present in the protein-ligand complex.

    Args:
        ring (str): A list of atom indices belonging to a ring that need to be modified.
        lig_index (int): An integer that is the first number of the ligand atom indices obtained from the protein-ligand, which is used to modify the ring indices

    Returns:
        list:  A new list with modified atom indicies.
    """
    return [atom_idx + lig_index for atom_idx in ring]


def process_pdb_file(input_pdb_filename):
    """Process a PDB file to make it compatible with the openmmdl_analysis package.

    Args:
        input_pdb_filename (str): path to the input PDB file
    """
    # Load the PDB file
    u = mda.Universe(input_pdb_filename)

    # Iterate through the topology to modify residue names
    for atom in u.atoms:
        resname = atom.resname

        # Check if the residue name is one of the specified water names
        if resname in ["SPC", "TIP3", "TIP4", "WAT", "T3P", "T4P", "T5P"]:
            atom.residue.resname = "HOH"
        elif resname == "*":
            atom.residue.resname = "UNK"

    # Save the modified topology to a new PDB file
    u.atoms.write(input_pdb_filename)


def extract_and_save_ligand_as_sdf(input_pdb_filename, output_filename, target_resname):
    """Extract and save the ligand from the receptor ligand complex PDB file into a new PDB file by itself.

    Args:
        input_pdb_filename (str): name of the input PDB file
        output_pdb_filename (str): name of the output SDF file
        target_resname (str): resname of the ligand in the original PDB file
    """
    # Load the PDB file using MDAnalysis
    u = mda.Universe(input_pdb_filename)

    # Find the ligand by its residue name
    ligand_atoms = u.select_atoms(f"resname {target_resname}")

    if len(ligand_atoms) == 0:
        print(f"No ligand with residue name '{target_resname}' found in the PDB file.")
        return

    #Using RDKit Converter to get SDF File
    lig_atoms = ligand_atoms.convert_to("RDKIT")

    #Write out the SDF file
    writer = SDWriter(output_filename)
    writer.write(lig_atoms)
    writer.close()


def renumber_atoms_in_residues(input_pdb_file, output_pdb_file, lig_name):
    """Renumer the atoms of the ligand in the topology PDB file.

    Args:
        input_pdb_file (str): Path to the initial PDB file.
        output_pdb_file (str): Path to the output PDB file.
        lig_name (str): Name of the ligand in the input PDB file.
    """
    # Read the input PDB file
    with open(input_pdb_file, "r") as f:
        pdb_lines = f.readlines()

    new_pdb_lines = []
    lig_residue_elements = {}

    for line in pdb_lines:
        if line.startswith("ATOM"):
            # Extract information from the ATOM line
            atom_serial = int(line[6:11])
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()

            # Check if the residue is residue_name
            if residue_name == lig_name:
                # Extract the element from the atom name (assuming it starts with a capital letter)
                element_match = re.match(r"([A-Z]+)", atom_name)
                if element_match:
                    element = element_match.group(1)
                else:
                    element = atom_name

                # Increment the count for the current element in the current residue_name residue
                lig_residue_elements[element] = lig_residue_elements.get(element, 0) + 1

                # Update the atom name based on the element count
                new_atom_name = f"{element}{lig_residue_elements[element]}"

                # Replace the old atom name with the new one
                line = line[:12] + f"{new_atom_name:4}" + line[16:]

        # Append the line to the new PDB lines
        new_pdb_lines.append(line)

    # Write the modified PDB lines to the output file
    with open(output_pdb_file, "w") as f:
        f.writelines(new_pdb_lines)


def replace_atom_type(data):
    """Replace wrong ligand atom types in the topology PDB file.

    Args:
        data (str): Text of the initial PDB file.

    Returns:
        str: Edited text of the PDB file.
    """
    lines = data.split("\n")
    for i, line in enumerate(lines):
        if " LIG  X" in line:
            # Extract the last column which contains the atom type (O/N/H)
            atom_type = line[12:13].strip()
            # Replace 'X' with the correct atom type
            lines[i] = line.replace(" LIG  X", f" LIG  {atom_type}")
    return "\n".join(lines)


def process_pdb(input_file, output_file):
    """Wrapper function to process a PDB file.

    Args:
        input_file (str): Path to the input PDB file.
        output_file (str): Path of the output PDB file.
    """
    with open(input_file, "r") as f:
        pdb_data = f.read()

    modified_data = replace_atom_type(pdb_data)

    with open(output_file, "w") as f:
        f.write(modified_data)
