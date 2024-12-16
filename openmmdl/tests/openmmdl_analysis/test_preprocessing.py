import pytest
import os
import MDAnalysis as mda
import mdtraj as md
from openmmdl.openmmdl_analysis.preprocessing import Preprocessing  # Adjust the import as needed


@pytest.fixture
def temp_pdb_file(tmpdir):
    """Fixture to create a temporary PDB file for input."""
    input_pdb = tmpdir.join("input.pdb")
    
    # Create synthetic PDB contents with various residue names, including water and unknown
    input_pdb_content = """
ATOM      1  O   WAT A   1      44.338  72.730  39.423  1.00  0.00           O
ATOM      2  O   WAT A   2      31.578  38.467  49.764  1.00  0.00           O
ATOM      3  O   WAT A   3      38.597  28.466  49.556  1.00  0.00           O
ATOM      4  O   WAT A   4      32.842  57.728  36.084  1.00  0.00           O
ATOM      5  O   WAT A   5      21.918  43.049  43.250  1.00  0.00           O
ATOM      6  O   WAT A   6      24.740  38.865  51.435  1.00  0.00           O
ATOM      7  O   WAT A   7      29.581  48.257  50.944  1.00  0.00           O
ATOM      8  O   WAT A   8      33.474  49.644  32.100  1.00  0.00           O
ATOM      9  O   WAT A   9      50.718  46.583  51.141  1.00  0.00           O
ATOM     10  O   *   A  10      35.529  47.920  25.052  1.00  0.00           O
ATOM     11  O   *   A  11      42.799  36.680  51.573  1.00  0.00           O
ATOM     12  O   *   A  12      44.499  25.396  38.351  1.00  0.00           O
ATOM     13  O   *   A  13      36.184  75.754  32.555  1.00  0.00           O
ATOM     14  O   *   A  14      28.279  70.291  27.450  1.00  0.00           O
    """
    
    # Write the contents to the file
    input_pdb.write(input_pdb_content)
    
    return str(input_pdb)

@pytest.fixture
def temp_pdb_lig_file(tmpdir):
    """Fixture to create a temporary PDB file with a ligand."""
    input_pdb = tmpdir.join("input.pdb")
    
    # Create synthetic PDB contents with a ligand (e.g., a simple water molecule)
    input_pdb_content = """
ATOM      1  N   LIG A   1      10.104  13.524   8.240  1.00 20.00           N  
ATOM      2  C   LIG A   1      11.104  14.524   8.240  1.00 20.00           C  
ATOM      3  O   LIG A   1      12.104  15.524   8.240  1.00 20.00           O  
ATOM      4  H   LIG A   1      13.104  16.524   8.240  1.00 20.00           H  
ATOM      5  H   LIG A   1      14.104  17.524   8.240  1.00 20.00           H  
ATOM      6  C   PRO A   2      15.104  18.524   8.240  1.00 20.00           C  
ATOM      7  N   PRO A   2      16.104  19.524   8.240  1.00 20.00           N  
ATOM      8  O   PRO A   2      17.104  20.524   8.240  1.00 20.00           O  
    """
    
    # Write the contents to the file
    input_pdb.write(input_pdb_content)
    
    return str(input_pdb)


def test_process_pdb_file(temp_pdb_file):
    input_pdb = temp_pdb_file
    
    # Create a Preprocessing instance
    preprocessing = Preprocessing()
    
    # Call the process_pdb_file method
    preprocessing.process_pdb_file(input_pdb)
    
    # Load the modified PDB file to check residue names
    u = mda.Universe(input_pdb)
    print(u)
    ag = u.select_atoms("all")
    for a in ag:
        print(a)
    
    # Check if the residues have been correctly renamed
    water_residues = [atom.residue.resname for atom in u.atoms if atom.residue.resname in ["HOH", "UNK"]]
    print(water_residues)
    
    # Check that the water residues were renamed correctly
    assert water_residues.count("HOH") == 9, "Expected 9 water residues (HOH)."
    assert water_residues.count("UNK") == 5, "Expected 5 unknown residues (UNK)."
    
    # Check the original residues were renamed (water and unknown)
    for atom in u.atoms:
        if atom.residue.resname in ["SPC", "TIP3", "TIP4", "WAT", "T3P", "T4P", "T5P"]:
            assert atom.residue.resname == "HOH", f"Expected 'HOH' but got {atom.residue.resname}"
        elif atom.residue.resname == "*":
            assert atom.residue.resname == "UNK", f"Expected 'UNK' but got {atom.residue.resname}"
    
    # Check if the output file exists and is not empty
    assert os.path.exists(input_pdb)
    assert os.path.getsize(input_pdb) > 0

def test_increase_ring_indices():
    # Create a Preprocessing instance
    preprocessing = Preprocessing()
    
    # Define test inputs
    ring = [1, 2, 3, 4, 5]  # Example atom indices in a ring
    lig_index = 10  # Ligand atom index to be added
    
    # Call the method
    result = preprocessing.increase_ring_indices(ring, lig_index)
    
    # Define the expected result
    expected_result = [11, 12, 13, 14, 15]  # The ring indices after adding lig_index
    
    # Assert the result is as expected
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

def test_extract_and_save_ligand_as_sdf(temp_pdb_lig_file, tmpdir):
    input_pdb = temp_pdb_lig_file
    output_sdf = tmpdir.join("output.sdf")
    target_resname = "LIG"  # The ligand residue name in the PDB file
    
    # Create a Preprocessing instance
    preprocessing = Preprocessing()
    
    # Call the method to extract the ligand and save it as SDF
    preprocessing.extract_and_save_ligand_as_sdf(input_pdb, str(output_sdf), target_resname)
    
    # Check if the output SDF file exists and is not empty
    assert os.path.exists(output_sdf)
    assert os.path.getsize(output_sdf) > 0, "The output SDF file is empty."
    
    # Check if the temporary PDB file (lig.pdb) was removed
    assert not os.path.exists("lig.pdb"), "Temporary PDB file 'lig.pdb' was not removed."
    

def test_renumber_atoms_in_residues(temp_pdb_lig_file, tmpdir):
    input_pdb = temp_pdb_lig_file
    output_pdb = tmpdir.join("output.pdb")
    lig_name = "LIG"  # The ligand residue name in the PDB file
    
    # Create a Preprocessing instance
    preprocessing = Preprocessing()
    
    # Call the method to renumber atoms in the ligand
    preprocessing.renumber_atoms_in_residues(input_pdb, str(output_pdb), lig_name)
    
    # Read the output PDB file
    with open(output_pdb, "r") as f:
        output_lines = f.readlines()

    # Check that the atoms of the ligand have been renumbered correctly
    renumbered_atoms = []
    for line in output_lines:
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            
            # Only check atoms of the ligand
            if residue_name == lig_name:
                renumbered_atoms.append(atom_name)
    
    # The atom names in the ligand should follow the pattern: N1, C1, O1, H1, H2
    expected_renumbered_atoms = ['N1', 'C1', 'O1', 'H1', 'H2']
    
    # Assert that the renumbered atom names match the expected output
    assert renumbered_atoms == expected_renumbered_atoms, f"Expected {expected_renumbered_atoms}, but got {renumbered_atoms}"
