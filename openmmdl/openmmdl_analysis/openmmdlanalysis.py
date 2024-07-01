"""
openmmdl_simulation.py
Perform Simulations of Protein-ligand complexes with OpenMM
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
import os
import argparse
import MDAnalysis as mda
import pandas as pd
import rdkit
import matplotlib
import pickle
import json
import Bio
import multiprocessing
import functools
from Bio import PDB
import cairosvg
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from plip.basic import config
from MDAnalysis.analysis import rms
from tqdm import tqdm

from openmmdl.openmmdl_analysis.preprocessing import (
    process_pdb_file,
    renumber_protein_residues,
    renumber_atoms_in_residues,
    process_pdb,
    extract_and_save_ligand_as_sdf,
)
from openmmdl.openmmdl_analysis.rmsd_calculation import (
    rmsd_for_atomgroups,
    RMSD_dist_frames,
)
from openmmdl.openmmdl_analysis.interaction_gathering import (
    process_trajectory,
    fill_missing_frames,
)
from openmmdl.openmmdl_analysis.binding_mode_processing import (
    gather_interactions,
    filtering_values,
    unique_data_generation,
    df_iteration_numbering,
    update_values,
    calculate_representative_frame,
    calculate_distance_matrix,
)
from openmmdl.openmmdl_analysis.markov_state_figure_generation import (
    min_transition_calculation,
    binding_site_markov_network,
)
from openmmdl.openmmdl_analysis.rdkit_figure_generation import (
    split_interaction_data,
    highlight_numbers,
    generate_interaction_dict,
    update_dict,
    create_and_merge_images,
    arranged_figure_generation,
    generate_ligand_image,
)
from openmmdl.openmmdl_analysis.barcode_generation import (
    barcodegeneration,
    plot_waterbridge_piechart,
    plot_barcodes_grouped,
)
from openmmdl.openmmdl_analysis.visualization_functions import (
    interacting_water_ids,
    save_interacting_waters_trajectory,
    cloud_json_generation,
)
from openmmdl.openmmdl_analysis.pml_writer import (
    generate_md_pharmacophore_cloudcenters,
    generate_bindingmode_pharmacophore,
    generate_pharmacophore_centers_all_points,
    generate_point_cloud_pml,
)
from openmmdl.openmmdl_analysis.find_stable_waters import (
    stable_waters_pipeline,
    analyze_protein_and_water_interaction,
)


def main():
    logo = "\n".join(
        [
            "     ,-----.    .-------.     .-''-.  ,---.   .--.,---.    ,---.,---.    ,---. ______       .---.      ",
            "   .'  .-,  '.  \  _(`)_ \  .'_ _   \ |    \  |  ||    \  /    ||    \  /    ||    _ `''.   | ,_|      ",
            "  / ,-.|  \ _ \ | (_ o._)| / ( ` )   '|  ,  \ |  ||  ,  \/  ,  ||  ,  \/  ,  || _ | ) _  \,-./  )      ",
            " ;  \  '_ /  | :|  (_,_) /. (_ o _)  ||  |\_ \|  ||  |\_   /|  ||  |\_   /|  ||( ''_'  ) |\  '_ '`)    ",
            " |  _`,/ \ _/  ||   '-.-' |  (_,_)___||  _( )_\  ||  _( )_/ |  ||  _( )_/ |  || . (_) `. | > (_)  )    ",
            " : (  '\_/ \   ;|   |     '  \   .---.| (_ o _)  || (_ o _) |  || (_ o _) |  ||(_    ._) '(  .  .-'    ",
            "  \ `_/  \  ) / |   |      \  `-'    /|  (_,_)\  ||  (_,_)  |  ||  (_,_)  |  ||  (_.\.' /  `-'`-'|___  ",
            "   '. \_/``'.'  /   )       \       / |  |    |  ||  |      |  ||  |      |  ||       .'    |        \ ",
            "     '-----'    `---'        `'-..-'  '--'    '--''--'      '--''--'      '--''-----'`      `--------` ",
            "              Prepare and Perform OpenMM Protein-Ligand MD Simulations                                 ",
            "                                     Alpha Version                                                     ",
        ]
    )

    parser = argparse.ArgumentParser(
        prog="openmmdl_analysis",
        description=logo,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t", dest="topology", help="Topology File after MD Simulation", required=True
    )
    parser.add_argument(
        "-d", dest="trajectory", help="Trajectory File in DCD Format", required=True
    )
    parser.add_argument(
        "-l", dest="ligand_sdf", help="Ligand in SDF Format", default=None
    )
    parser.add_argument(
        "-n",
        dest="ligand_name",
        help="Ligand Name (3 Letter Code in PDB)",
        default=None,
    )
    parser.add_argument(
        "-b",
        dest="binding",
        help="Binding Mode Treshold for Binding Mode in %%",
        default=40,
    )
    parser.add_argument(
        "-df",
        dest="dataframe",
        help='Dataframe (use if the interactions were already calculated, default name would be "df_all.csv")',
        default=None,
    )
    parser.add_argument(
        "-m",
        dest="min_transition",
        help="Minimal Transition percentage for Markov State Model",
        default=1,
    )
    parser.add_argument(
        "-c",
        dest="cpu_count",
        help="cores, specify how many PC cores should be used, default is half of the PC cores",
        default=os.cpu_count() // 2,
    )
    parser.add_argument(
        "-p",
        dest="generate_pml",
        help="Generate .pml files for pharmacophore visualization",
        default=False,
    )
    parser.add_argument(
        "-r",
        dest="frame_rmsd",
        help='RMSD Difference between frames calculation type "True" to use it default is False,',
        default="No",
    )
    parser.add_argument(
        "-nuc",
        dest="receptor_nucleic",
        help="Treat nucleic acids as receptor",
        default=False,
    )
    parser.add_argument(
        "-s",
        dest="special_ligand",
        help="Calculate interactions with special ligands",
        default=None,
    )
    parser.add_argument(
        "-pep",
        dest="peptide",
        help="Calculate interactions with peptides. Give the peptides chain name as input. Defaults to None",
        default=None,
    )
    parser.add_argument(
        "-ref",
        dest="reference",
        help="Add a reference PDB to renumber the residue numbers",
        default=None,
    )
    parser.add_argument(
        "-w",
        dest="stable_water_analysis",
        help="Should stable water analysis be performed? True or False",
        default=False,
    )
    parser.add_argument(
        "-rep",
        dest="representative_frame",
        help="Calculate the representative frame for each binding mode. Defaults to False",
        default=False,
    )

    parser.add_argument(
        "--watereps",
        dest="water_eps",
        help="Set the Eps for clustering, this defines how big clusters can be spatially in Angstrom",
        default=1.0,
    )
    
    parser.add_argument(
        "--figure",
        dest="figure_type",
        help="File type for the figures, default is png. Can be changed to all file types supported by matplotlib.",
        default="png",
    )

    pdb_md = None
    input_formats = [
        ".pdb",
        ".dcd",
        ".sdf",
        ".csv",
        ".tpr",
        ".xtc",
        "trr",
    ]
    args = parser.parse_args()
    if input_formats[0] not in args.topology and input_formats[4] not in args.topology:
        print("Topology is missing, try the absolute path")
    if (
        input_formats[1] not in args.trajectory
        and input_formats[5] not in args.trajectory
        and input_formats[6] not in args.trajectory
    ):
        print("Trajectory is missing, try the absolute path")

    # set variables for analysis and preprocess input files
    topology = args.topology
    trajectory = args.trajectory
    # enable gromacs support and write topology and trajectory files
    if ".tpr" in args.topology and (
        ".xtc" in args.trajectory or ".trr" in args.trajectory
    ):
        print("\033[1mGromacs format detected. Writing compatible file formats.\033[0m")
        u = mda.Universe(args.topology, args.trajectory)
        with mda.Writer("trajectory.dcd", n_atoms=u.atoms.n_atoms) as W:
            first_frame_saved = False
            for ts in u.trajectory:
                if not first_frame_saved:
                    with mda.Writer(
                        "topology.pdb", n_atoms=u.atoms.n_atoms
                    ) as pdb_writer:
                        pdb_writer.write(u.atoms)
                        first_frame_saved = True
                W.write(u.atoms)
        pdb_md = mda.Universe("topology.pdb", "trajectory.dcd")
        topology = "topology.pdb"
        trajectory = "trajectory.dcd"
    water_eps = float(args.water_eps)
    stable_water_analysis = bool(args.stable_water_analysis)

    # The following is the current water analysis if no ligand is present.
    if not args.ligand_sdf and args.peptide == None and stable_water_analysis:
        print("All analyses will be run which can be done without a ligand present")
        # ...
        stable_waters_pipeline(topology, trajectory, water_eps)
        analyze_protein_and_water_interaction(
            topology, "representative_waters.pdb", water_eps
        )

    # if input_formats[2] not in args.ligand_sdf:
    #     print("SDF is missing, try the absolute path. Maybe you don't own a ligand (sad), in this case we'll only analyze the stable waters!")
    # if args.ligand_name == None:
    #     print("Ligand Name is Missing, Add Ligand Name")

    # set variables for analysis and preprocess input files
    ligand_sdf = args.ligand_sdf
    ligand = args.ligand_name
    frame_rmsd = args.frame_rmsd
    if ligand == "*":
        ligand = "UNK"
    treshold = int(args.binding)
    dataframe = args.dataframe
    min_transition = int(args.min_transition)
    cpu_count = int(args.cpu_count)
    generate_pml = bool(args.generate_pml)
    receptor_nucleic = bool(args.receptor_nucleic)
    special_ligand = args.special_ligand
    reference = args.reference
    peptide = args.peptide
    fig_type = args.figure_type

    generate_representative_frame = args.representative_frame

    if reference != None:
        print("\033[1mPDB File residues are being renumbered\033[0m")
        renumber_protein_residues(topology, reference, topology)
    process_pdb_file(topology)
    print("\033[1mFiles are preprocessed\033[0m")

    if ligand_sdf == None:
        extract_and_save_ligand_as_sdf(topology, "./ligand_prepared.sdf", ligand)
        ligand_sdf = "./ligand_prepared.sdf"

    if not pdb_md:
        pdb_md = mda.Universe(topology, trajectory)

    # Writing out the complex of the protein and ligand with water around 10A of the ligand
    complex = pdb_md.select_atoms(
        f"protein or nucleic or resname {ligand} or (resname HOH and around 10 resname {ligand}) or resname {special_ligand}"
    )
    complex.write("complex.pdb")
    renumber_atoms_in_residues("complex.pdb", "complex.pdb", ligand)
    process_pdb("complex.pdb", "complex.pdb")
    # Writing out the ligand in a separate pdb file for ring calculation
    if peptide == None:
        ligand_complex = pdb_md.select_atoms(f"resname {ligand}")
        ligand_complex_no_h = pdb_md.select_atoms(f"resname {ligand} and not (name H*)")
        if special_ligand != None:
            ligand_special = pdb_md.select_atoms(
                f"resname {ligand} or resname {special_ligand}"
            )
            ligand_special.write("ligand_special.pdb")
        ligand_complex_no_h.write("lig_no_h.pdb")
        renumber_atoms_in_residues("lig_no_h.pdb", "lig_no_h.pdb", ligand)
        process_pdb("lig_no_h.pdb", "lig_no_h.pdb")
        ligand_complex.write("lig.pdb")

        # getting Ring Information from the ligand pdb file
        lig_rd = rdkit.Chem.rdmolfiles.MolFromPDBFile("lig.pdb")
        try:
            lig_rd_ring = lig_rd.GetRingInfo()
        except AttributeError:
            print("\033[1mCould not get the ring information.\033[0m")
            print(
                "\033[1mTry to remove lone pairs prior to running an analysis!\033[0m"
            )
            exit()

        # getting the index of the first atom of the ligand from the complex pdb
        novel_complex = mda.Universe("complex.pdb")
        novel_complex_ligand = novel_complex.select_atoms(f"resname {ligand}")
        for atom in novel_complex_ligand:
            lig_index = atom.id
            break
        ligand_rings = []
        ligand_no_hydrogens = mda.Universe("lig_no_h.pdb")
        lig_no_hydrogens = ligand_no_hydrogens.select_atoms("all")
        complex_universe = mda.Universe("complex.pdb")
        complex_lig = novel_complex.select_atoms(f"resname {ligand}")

        # Iterate through each ring, increase indices by 1, and print the updated rings
        for atom_ring in lig_rd_ring.AtomRings():
            current_ring = []
            for atom_ring_single in atom_ring:
                for ligand_ring_atom in lig_no_hydrogens:
                    if atom_ring_single + 1 == ligand_ring_atom.id:
                        # print(ligand_ring_atom.name)
                        for complex_lig_atom in novel_complex_ligand:
                            if ligand_ring_atom.name == complex_lig_atom.name:
                                true_number = complex_lig_atom.id
                                updated_ring_2 = true_number
                                current_ring.append(updated_ring_2)
            ligand_rings.append(current_ring)
        print("\033[1mLigand ring data gathered\033[0m")

    if peptide != None:
        ligand_rings = None

    os.makedirs("RMSD", exist_ok=True)
    if receptor_nucleic:
        rmsd_for_atomgroups(
            f"{topology}",
            f"{trajectory}",
            fig_type,
            selection1="nucleicbackbone",
            selection2=["nucleic", f"resname {ligand}"],
        )
        if frame_rmsd != "No":
            RMSD_dist_frames(
                f"{topology}", f"{trajectory}", fig_type, lig=f"{ligand}", nucleic=True
            )
            print("\033[1mRMSD calculated\033[0m")
    elif peptide != None:
        rmsd_for_atomgroups(
            f"{topology}",
            f"{trajectory}",
            fig_type,
            selection1="backbone",
            selection2=["protein", f"chainID {peptide}"],
        )
        if frame_rmsd != "No":
            RMSD_dist_frames(f"{topology}", f"{trajectory}", fig_type, lig=f"chainID {peptide}")
            print("\033[1mRMSD calculated\033[0m")
    else:
        rmsd_for_atomgroups(
            f"{topology}",
            f"{trajectory}",
            fig_type,
            selection1="backbone",
            selection2=["protein", f"resname {ligand}"],
        )
        if frame_rmsd != "No":
            RMSD_dist_frames(f"{topology}", f"{trajectory}", fig_type, lig=f"{ligand}")
            print("\033[1mRMSD calculated\033[0m")

    if receptor_nucleic:
        config.DNARECEPTOR = True
    if peptide != None:
        config.PEPTIDES = [peptide]

    interaction_list = pd.DataFrame(
        columns=[
            "RESNR",
            "RESTYPE",
            "RESCHAIN",
            "RESNR_LIG",
            "RESTYPE_LIG",
            "RESCHAIN_LIG",
            "DIST",
            "LIGCARBONIDX",
            "PROTCARBONIDX",
            "LIGCOO",
            "PROTCOO",
        ]
    )

    interaction_list = process_trajectory(
        pdb_md,
        dataframe=dataframe,
        num_processes=cpu_count,
        lig_name=ligand,
        special_ligand=special_ligand,
        peptide=peptide,
    )

    interaction_list["Prot_partner"] = (
        interaction_list["RESNR"].astype(str)
        + interaction_list["RESTYPE"]
        + interaction_list["RESCHAIN"]
    )

    interaction_list = fill_missing_frames(
        interaction_list, md_len=len(pdb_md.trajectory) - 1
    )

    interaction_list.to_csv("missing_frames_filled.csv")

    interaction_list = interaction_list.reset_index(drop=True)

    unique_columns_rings_grouped = gather_interactions(
        interaction_list, ligand_rings, peptide=peptide
    )

    interactions_all = interaction_list.copy()

    # Add Frames + Treshold by user
    filtered_values = filtering_values(
        threshold=treshold / 100,
        frames=len(pdb_md.trajectory) - 1,
        df=interaction_list,
        unique_columns_rings_grouped=unique_columns_rings_grouped,
    )

    filtering_all = filtering_values(
        threshold=0.00001,
        frames=len(pdb_md.trajectory) - 1,
        df=interactions_all,
        unique_columns_rings_grouped=unique_columns_rings_grouped,
    )

    # Replace NaN values with 0 in the entire DataFrame
    interaction_list.fillna(0, inplace=True)
    interactions_all.fillna(0, inplace=True)

    unique_data = unique_data_generation(filtered_values)
    unique_data_all = unique_data_generation(filtering_all)

    # Iteration through the dataframe and numbering the interactions with 1 and 0, depending if the interaction exists or not
    df_iteration_numbering(interaction_list, unique_data, peptide=peptide)
    df_iteration_numbering(interactions_all, unique_data_all, peptide=peptide)
    print("\033[1mInteraction values assigned\033[0m")

    # Saving the dataframe
    interactions_all.to_csv("df_all.csv")

    # Group by 'FRAME' and transform each group to set all values to 1 if there is at least one 1 in each column
    grouped_frames_treshold = interaction_list.groupby("FRAME", as_index=False)[
        list(unique_data.values())
    ].max()
    grouped_frames_treshold = grouped_frames_treshold.set_index("FRAME", drop=False)

    update_values(interaction_list, grouped_frames_treshold, unique_data)

    # Change the FRAME column value type to int
    grouped_frames_treshold["FRAME"] = grouped_frames_treshold["FRAME"].astype(int)

    # Extract all columns except 'FRAME' and the index column
    selected_columns = grouped_frames_treshold.columns[1:-1]

    # Create a list of lists with the values from selected columns for each row
    treshold_result_list = [
        row[selected_columns].values.tolist()
        for _, row in grouped_frames_treshold.iterrows()
    ]

    # Calculate the occurrences of each list in the result_list
    treshold_occurrences = Counter(tuple(lst) for lst in treshold_result_list)

    # Create a new column 'fingerprint' in the DataFrame
    grouped_frames_treshold["fingerprint"] = None

    # Set the 'fingerprint' column values based on the corresponding index in result_list
    for index, fingerprint_value in enumerate(treshold_result_list, 1):
        grouped_frames_treshold.at[index, "fingerprint"] = fingerprint_value

    # Assuming your original DataFrame is named 'df'
    # First, we'll create a new column 'Binding_fingerprint_hbond'
    grouped_frames_treshold["Binding_fingerprint_treshold"] = ""

    # Dictionary to keep track of encountered fingerprints and their corresponding labels
    treshold_fingerprint_dict = {}

    # Counter to generate the labels (Hbond_Binding_1, Hbond_Binding_2, etc.)
    label_counter = 1

    # Iterate through the rows and process the 'fingerprint' column
    for index, row in grouped_frames_treshold.iterrows():
        fingerprint = tuple(row["fingerprint"])

        # Check if the fingerprint has been encountered before
        if fingerprint in treshold_fingerprint_dict:
            grouped_frames_treshold.at[
                index, "Binding_fingerprint_treshold"
            ] = treshold_fingerprint_dict[fingerprint]
        else:
            # Assign a new label if the fingerprint is new
            label = f"Binding_Mode_{label_counter}"
            treshold_fingerprint_dict[fingerprint] = label
            grouped_frames_treshold.at[index, "Binding_fingerprint_treshold"] = label
            label_counter += 1

    # Group the DataFrame by the 'Binding_fingerprint_hbond' column and create the dictionary of the fingerprints
    fingerprint_dict = grouped_frames_treshold["Binding_fingerprint_treshold"].to_dict()
    combined_dict = {"all": []}
    for key, value in fingerprint_dict.items():
        combined_dict["all"].append(value)

    # Generate Markov state figures of the binding modes
    total_frames = len(pdb_md.trajectory) - 1
    min_transitions = min_transition_calculation(min_transition)
    binding_site_markov_network(total_frames, min_transitions, combined_dict, fig_type)
    print("\033[1mMarkov State Figure generated\033[0m")

    # Get the top 10 nodes with the most occurrences
    node_occurrences = {
        node: combined_dict["all"].count(node) for node in set(combined_dict["all"])
    }
    top_10_nodes = sorted(node_occurrences, key=node_occurrences.get, reverse=True)[:10]
    top_10_nodes_with_occurrences = {
        node: node_occurrences[node] for node in top_10_nodes
    }

    # Initialize an empty dictionary to store the result
    columns_with_value_1 = {}

    for treshold, row_count in top_10_nodes_with_occurrences.items():
        for i in range(1, row_count + 1):
            # Get the row corresponding to the treshold value
            row = grouped_frames_treshold.loc[
                grouped_frames_treshold["Binding_fingerprint_treshold"] == treshold
            ].iloc[i - 1]

            # Extract column names with value 1 in that row
            columns_with_1 = row[row == 1].index.tolist()

            # Convert the list to a set to remove duplicates
            columns_set = set(columns_with_1)

            # Add the columns to the dictionary under the corresponding threshold
            if treshold not in columns_with_value_1:
                columns_with_value_1[treshold] = set()
            columns_with_value_1[treshold].update(columns_set)

    # Generate an Figure for each of the binding modes with rdkit Drawer with the atoms interacting highlighted by colors
    try:
        if peptide is None:
            matplotlib.use("Agg")
            binding_site = {}
            merged_image_paths = []
            for binding_mode, values in columns_with_value_1.items():
                binding_site[binding_mode] = values
                occurrence_count = top_10_nodes_with_occurrences[binding_mode]
                occurrence_percent = 100 * occurrence_count / total_frames
                # Convert ligand to RDKit with Converter
                lig_atoms = complex_lig.convert_to("RDKIT")
                # Remove Hydrogens and get 2D representation
                prepared_ligand = Chem.RemoveAllHs(lig_atoms)
                AllChem.Compute2DCoords(prepared_ligand)
                split_data = split_interaction_data(values)
                # Get the highlighted atom indices based on interaction type
                (
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
                ) = highlight_numbers(split_data, starting_idx=lig_index)

                # Generate a dictionary for hydrogen bond acceptors
                hbond_acceptor_dict = generate_interaction_dict(
                    "hbond_acceptor", highlighted_hbond_acceptor
                )
                # Generate a dictionary for hydrogen bond acceptors and donors
                hbond_both_dict = generate_interaction_dict(
                    "hbond_both", highlighted_hbond_both
                )
                # Generate a dictionary for hydrogen bond donors
                hbond_donor_dict = generate_interaction_dict(
                    "hbond_donor", highlighted_hbond_donor
                )
                # Generate a dictionary for hydrophobic features
                hydrophobic_dict = generate_interaction_dict(
                    "hydrophobic", highlighted_hydrophobic
                )
                # Generate a dictionary for water bridge interactions
                waterbridge_dict = generate_interaction_dict(
                    "waterbridge", highlighted_waterbridge
                )
                # Generate a dictionary for pistacking
                pistacking_dict = generate_interaction_dict(
                    "pistacking", highlighted_pistacking
                )
                # Generate a dictionary for halogen interactions
                halogen_dict = generate_interaction_dict("halogen", highlighted_halogen)
                # Generate a dictionary for negative ionizables
                ni_dict = generate_interaction_dict("ni", highlighted_ni)
                # Generate a dictionary for negative ionizables
                pi_dict = generate_interaction_dict("pi", highlighted_pi)
                # Generate a dictionary for pication
                pication_dict = generate_interaction_dict(
                    "pication", highlighted_pication
                )
                # Generate a dictionary for metal interactions
                metal_dict = generate_interaction_dict("metal", highlighted_metal)

                # Call the function to update hbond_donor_dict with values from other dictionaries
                update_dict(
                    hbond_donor_dict,
                    hbond_acceptor_dict,
                    ni_dict,
                    pi_dict,
                    hydrophobic_dict,
                    hbond_both_dict,
                    waterbridge_dict,
                    pistacking_dict,
                    halogen_dict,
                    pication_dict,
                    metal_dict,
                )

                # Convert the highlight_atoms to int type for rdkit drawer
                highlight_atoms = [
                    int(x)
                    for x in highlighted_hbond_donor
                    + highlighted_hbond_acceptor
                    + highlighted_hbond_both
                    + highlighted_ni
                    + highlighted_pi
                    + highlighted_hydrophobic
                    + highlighted_waterbridge
                    + highlighted_pistacking
                    + highlighted_halogen
                    + highlighted_pication
                    + highlighted_metal
                ]
                highlight_atoms = list(set(highlight_atoms))

                # Convert the RDKit molecule to SVG format with atom highlights
                drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
                drawer.DrawMolecule(
                    prepared_ligand,
                    highlightAtoms=highlight_atoms,
                    highlightAtomColors=hbond_donor_dict,
                )
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText().replace("svg:", "")

                # Save the SVG to a file
                with open(f"{binding_mode}.svg", "w") as f:
                    f.write(svg)

                # Convert the svg to an png
                cairosvg.svg2png(
                    url=f"{binding_mode}.svg", write_to=f"{binding_mode}.png"
                )

                # Generate the interactions legend and combine it with the ligand png
                merged_image_paths = create_and_merge_images(
                    binding_mode, occurrence_percent, split_data, merged_image_paths
                )

            # Create Figure with all Binding modes
            arranged_figure_generation(
                merged_image_paths, "all_binding_modes_arranged.png"
            )
            generate_ligand_image(
                ligand, "complex.pdb", "lig_no_h.pdb", f"ligand_numbering.svg"
            )
            if fig_type == "png":
                cairosvg.svg2png(
                    url=f"ligand_numbering.svg", write_to=f"ligand_numbering.png"
                )
            print("\033[1mBinding mode figure generated\033[0m")
    except Exception as e:
        print(f"Ligand could not be recognized, use the -l option")

    df_all = pd.read_csv("df_all.csv")

    # get the top 10 bindingmodes with the most occurrences
    binding_modes = grouped_frames_treshold["Binding_fingerprint_treshold"].str.split(
        "\n"
    )
    all_binding_modes = [mode.strip() for sublist in binding_modes for mode in sublist]
    binding_mode_counts = pd.Series(all_binding_modes).value_counts()
    top_10_binding_modes = binding_mode_counts.head(10)
    total_binding_modes = len(all_binding_modes)
    result_dict = {
        "Binding Mode": [],
        "First Frame": [],
        "All Frames": [],
        "Representative Frame": [],
        "Percentage Occurrence": [],
    }
    if generate_representative_frame:
        DM = calculate_distance_matrix(
            pdb_md,
            f"protein or nucleic or resname {ligand} or resname {special_ligand}",
        )
        modes_to_process = top_10_binding_modes.index
        for mode in tqdm(modes_to_process):
            result_dict["Binding Mode"].append(mode)
            first_frame = grouped_frames_treshold.loc[
                grouped_frames_treshold["Binding_fingerprint_treshold"].str.contains(
                    mode
                ),
                "FRAME",
            ].iloc[0]
            all_frames = grouped_frames_treshold.loc[
                grouped_frames_treshold["Binding_fingerprint_treshold"].str.contains(
                    mode
                ),
                "FRAME",
            ].tolist()
            percent_occurrence = (
                top_10_binding_modes[mode] / total_binding_modes
            ) * 100
            result_dict["First Frame"].append(first_frame)
            result_dict["All Frames"].append(all_frames)
            result_dict["Percentage Occurrence"].append(percent_occurrence)
            representative_frame = calculate_representative_frame(all_frames, DM)
            result_dict["Representative Frame"].append(representative_frame)
        top_10_binding_modes_df = pd.DataFrame(result_dict)
        top_10_binding_modes_df.to_csv("top_10_binding_modes.csv")
        print("\033[1mFound representative frame for each binding mode\033[0m")
        # save bindingmode pdb and .pml
        id_num = 0
        for index, row in top_10_binding_modes_df.iterrows():
            b_mode = row["Binding Mode"]
            rep_frame = int(row["Representative Frame"])
            pdb_md.trajectory[rep_frame]
            frame_atomgroup = pdb_md.atoms
            frame_atomgroup.write(f"./Binding_Modes_Markov_States/{b_mode}.pdb")
            if generate_pml:
                filtered_df_all = df_all[df_all["FRAME"] == rep_frame]
                filtered_df_bindingmodes = grouped_frames_treshold[
                    grouped_frames_treshold["FRAME"] == rep_frame
                ]
                bindingmode_dict = {}
                for index, row in filtered_df_bindingmodes.iterrows():
                    for column in filtered_df_bindingmodes.columns:
                        if column not in [
                            "FRAME",
                            "FRAME.1",
                            "fingerprint",
                            "Binding_fingerprint_treshold",
                        ]:
                            if row[column] == 1:
                                if column not in bindingmode_dict:
                                    bindingmode_dict[column] = {
                                        "LIGCOO": [],
                                        "PROTCOO": [],
                                    }  # Initialize a nested dictionary for each key if not already present
                                for index2, row2 in filtered_df_all.iterrows():
                                    if row2[column] == 1:
                                        # Extract the string representation of the tuple
                                        tuple_string = row2["LIGCOO"]
                                        # Split the string into individual values using a comma as the delimiter
                                        ligcoo_values = tuple_string.strip("()").split(
                                            ","
                                        )
                                        # Convert the string values to float
                                        ligcoo_values = [
                                            float(value.strip())
                                            for value in ligcoo_values
                                        ]

                                        # Extract the string representation of the tuple for PROTCOO
                                        tuple_string = row2["PROTCOO"]
                                        # Split the string into individual values using a comma as the delimiter
                                        protcoo_values = tuple_string.strip("()").split(
                                            ","
                                        )
                                        # Convert the string values to float
                                        protcoo_values = [
                                            float(value.strip())
                                            for value in protcoo_values
                                        ]

                                        bindingmode_dict[column]["LIGCOO"].append(
                                            ligcoo_values
                                        )
                                        bindingmode_dict[column]["PROTCOO"].append(
                                            protcoo_values
                                        )
                generate_bindingmode_pharmacophore(
                    bindingmode_dict, ligand, f"{ligand}_complex", b_mode, id_num
                )

        print("\033[1mBinding mode pdb files saved\033[0m")

    hydrophobic_interactions = df_all.filter(regex="hydrophobic").columns
    acceptor_interactions = df_all.filter(regex="Acceptor_hbond").columns
    donor_interactions = df_all.filter(regex="Donor_hbond").columns
    pistacking_interactions = df_all.filter(regex="pistacking").columns
    halogen_interactions = df_all.filter(regex="halogen").columns
    waterbridge_interactions = df_all.filter(regex="waterbridge").columns
    pication_interactions = df_all.filter(regex="pication").columns
    saltbridge_ni_interactions = df_all.filter(regex="NI_saltbridge").columns
    saltbridge_pi_interactions = df_all.filter(regex="PI_saltbridge").columns
    metal_interactions = df_all.filter(regex="metal").columns

    waterbridge_barcodes = {}
    for waterbridge_interaction in waterbridge_interactions:
        barcode = barcodegeneration(df_all, waterbridge_interaction)
        waterbridge_barcodes[waterbridge_interaction] = barcode

    interaction_types = {
        "hydrophobic": hydrophobic_interactions,
        "acceptor": acceptor_interactions,
        "donor": donor_interactions,
        "pistacking": pistacking_interactions,
        "halogen": halogen_interactions,
        "waterbridge": waterbridge_interactions,
        "pication": pication_interactions,
        "saltbridge_ni": saltbridge_ni_interactions,
        "saltbridge_pi": saltbridge_pi_interactions,
        "metal": metal_interactions,
    }

    for interaction_type, interaction_data in interaction_types.items():
        plot_barcodes_grouped(interaction_data, df_all, interaction_type, fig_type)

    plot_waterbridge_piechart(df_all, waterbridge_barcodes, waterbridge_interactions, fig_type)
    print("\033[1mBarcodes generated\033[0m")

    interacting_water_id_list = interacting_water_ids(df_all, waterbridge_interactions)

    # dump interacting waters for visualization
    os.makedirs("Visualization", exist_ok=True)  # Create the folder if it doesn't exist
    with open("Visualization/interacting_waters.pkl", "wb") as f:
        pickle.dump(interacting_water_id_list, f)
    save_interacting_waters_trajectory(
        topology, trajectory, interacting_water_id_list, ligand, special_ligand
    )

    # save clouds for visualization with NGL
    with open("Visualization/clouds.json", "w") as f:
        json.dump(cloud_json_generation(df_all), f)

    # generate poincloud pml for visualization
    cloud_dict = {}
    cloud_dict["H"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="hydrophobic").columns
    )
    cloud_dict["HBA"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="Acceptor_hbond").columns
    )
    cloud_dict["HBD"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="Donor_hbond").columns
    )
    cloud_dict["AR"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="pistacking").columns
    )
    cloud_dict["PI"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="PI_saltbridge").columns
    )
    cloud_dict["NI"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="NI_saltbridge").columns
    )
    cloud_dict["M"] = generate_pharmacophore_centers_all_points(
        df_all, df_all.filter(regex="metal").columns
    )

    if generate_pml:
        generate_point_cloud_pml(cloud_dict, f"{ligand}_complex", "point_cloud")

        # generate combo pharmacophore of the md with each interaction as a single pharmacophore feature
        generate_md_pharmacophore_cloudcenters(
            df_all, ligand, "combopharm.pml", f"{ligand}_complex"
        )
        print("\033[1mPharmacophores generated\033[0m")

    print("\033[1mAnalysis is Finished.\033[0m")

    if stable_water_analysis:
        stable_waters_pipeline(topology, trajectory, water_eps)
        analyze_protein_and_water_interaction(
            topology, "representative_waters.pdb", water_eps
        )


if __name__ == "__main__":
    main()
