import numpy as np
import pandas as pd
import re
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pytest
from openmmdl.openmmdl_analysis.barcode_generation import BarcodeGenerator, BarcodePlotter

# Barcode generation tests
@pytest.fixture
def sample_dataframe_barcode_generation():
    data = {
        'FRAME': [1, 1, 2, 2, 3],
        'Interaction1': [1, 0, 1, 0, 0],
        'Interaction2': [0, 0, 0, 1, 1],
        'WATER_IDX': [101, 102, 103, 104, 105],
    }
    return pd.DataFrame(data)

def test_barcodegeneration(sample_dataframe_barcode_generation):
    interaction = 'Interaction1'
    barcode_generator = BarcodeGenerator(sample_dataframe_barcode_generation)

    barcode = barcode_generator.generate_barcode(interaction)
    
    assert isinstance(barcode, np.ndarray)
    
    expected_barcode = np.array([1, 1, 0])
    assert np.array_equal(barcode, expected_barcode)
    
def test_waterids_barcode_generator(sample_dataframe_barcode_generation):
    interaction = 'Interaction2'
    barcode_generator = BarcodeGenerator(sample_dataframe_barcode_generation)
    waterid_barcode = barcode_generator.generate_waterids_barcode(interaction)
    
    # Test if the output is a list
    assert isinstance(waterid_barcode, list)
    
    # Test the expected waterid barcode for the sample dataframe and interaction
    expected_waterid_barcode = [0, 104, 105]
    assert waterid_barcode == expected_waterid_barcode

def test_plot_barcodes(tmp_path):
    df_all = pd.DataFrame()
    barcode_plotter = BarcodePlotter(df_all)

    # Test case 1: No barcode
    save_path = tmp_path / "no_barcodes.png"
    barcode_plotter.plot_barcodes({}, save_path)
    assert not save_path.exists()

    # Test case 2: Single barcode
    barcode_data = {
        'Barcode1': np.array([1, 1, 0, 1, 0]),
    }
    save_path = tmp_path / "single_barcode.png"
    barcode_plotter.plot_barcodes(barcode_data, save_path)
    assert save_path.exists()

    # Test case 3: Multiple barcodes
    barcode_data = {
        "Barcode 1": np.array([1, 0, 1, 0, 1, 0]),
        "Barcode 2": np.array([0, 1, 0, 1, 0, 1]),
    }
    save_path = tmp_path / "multiple_barcodes.png"
    barcode_plotter.plot_barcodes(barcode_data, save_path)
    assert save_path.exists()

def test_plot_waterbridge_piechart(sample_dataframe_barcode_generation):
    barcode_plotter = BarcodePlotter(sample_dataframe_barcode_generation)
    waterbridge_barcodes = [np.array([1, 0, 1, 0]), np.array([0, 0, 0, 1])]
    waterbridge_interactions = ['Interaction1', 'Interaction2']
    fig_type = 'png'

    # Run the plotting function
    barcode_plotter.plot_waterbridge_piechart(waterbridge_barcodes, waterbridge_interactions, fig_type)

    # Check if files are saved in the hardcoded directory
    output_dir = os.path.join("Barcodes", "Waterbridge_Piecharts")
    for interaction in waterbridge_interactions:
        outname_png = os.path.join(output_dir, f"{interaction}.{fig_type}")
        assert os.path.exists(outname_png), f"File not found: {outname_png}"

def test_plot_barcodes_grouped():
    df_all = pd.DataFrame({
        'FRAME': [0, 1, 2],
        'atom1_atom2_interaction': [1, 0, 1],
        'atom3_atom4_interaction': [0, 1, 1],
    })

    interactions = ['atom1_atom2_interaction', 'atom3_atom4_interaction']
    interaction_type = 'interaction'
    fig_type = 'png'
    barcode_plotter = BarcodePlotter(df_all)

    barcode_plotter.plot_barcodes_grouped(interactions, interaction_type, fig_type)

    # Expected file paths
    atom2_dir = Path("Barcodes/atom2")
    atom4_dir = Path("Barcodes/atom4")
    total_path = Path(f"Barcodes/{interaction_type}_interactions.{fig_type}")

    # Validate that directories and files were created
    assert atom2_dir.exists(), f"Directory not found: {atom2_dir}"
    assert atom4_dir.exists(), f"Directory not found: {atom4_dir}"

    atom2_file = atom2_dir / f"atom2_{interaction_type}_barcodes.{fig_type}"
    atom4_file = atom4_dir / f"atom4_{interaction_type}_barcodes.{fig_type}"

    assert atom2_file.exists(), f"File not found: {atom2_file}"
    assert atom4_file.exists(), f"File not found: {atom4_file}"
    assert total_path.exists(), f"File not found: {total_path}"
