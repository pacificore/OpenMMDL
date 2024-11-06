import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm
import os
import MDAnalysis as mda
from MDAnalysis.analysis import rms, diffusionmap
from MDAnalysis.analysis.distances import dist
from typing import Optional, List, Tuple, Union


@jit(nopython=True, parallel=True, nogil=True)
def calc_rmsd_2frames_jit(ref: np.ndarray, frame: np.ndarray) -> float:
    dist = np.zeros(len(frame))
    for atom in range(len(frame)):
        dist[atom] = (
            (ref[atom][0] - frame[atom][0]) ** 2
            + (ref[atom][1] - frame[atom][1]) ** 2
            + (ref[atom][2] - frame[atom][2]) ** 2
        )

    return np.sqrt(dist.mean())


class RMSDAnalyzer:
    def __init__(self, prot_lig_top_file: str, prot_lig_traj_file: str) -> None:
        self.prot_lig_top_file: str = prot_lig_top_file
        self.prot_lig_traj_file: str = prot_lig_traj_file
        self.universe: mda.Universe = mda.Universe(
            prot_lig_top_file, prot_lig_traj_file
        )

    def rmsd_for_atomgroups(
        self, fig_type: str, selection1: str, selection2: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate the RMSD for selected atom groups, and save the csv file and plot.

        Args:
            fig_type (str): Type of the figure to save (e.g., 'png', 'jpg').
            selection1 (str): Selection string for main atom group, also used during alignment.
            selection2 (list, optional): Selection strings for additional atom groups. Defaults to None.

        Returns:
            pandas dataframe: rmsd_df. DataFrame containing RMSD of the selected atom groups over time.
        """
        self.universe.trajectory[0]
        ref = self.universe
        rmsd_analysis = rms.RMSD(
            self.universe, ref, select=selection1, groupselections=selection2
        )
        rmsd_analysis.run()
        columns = [selection1, *selection2] if selection2 else [selection1]
        rmsd_df = pd.DataFrame(np.round(rmsd_analysis.rmsd[:, 2:], 2), columns=columns)
        rmsd_df.index.name = "frame"

        # Create the directory if it doesn't exist
        output_directory = "./RMSD/"
        os.makedirs(output_directory, exist_ok=True)

        # Save the RMSD values to a CSV file in the created directory
        rmsd_df.to_csv(f"{output_directory}/RMSD_over_time.csv", sep=" ")

        # Plot and save the RMSD over time as a PNG file
        rmsd_df.plot(title="RMSD of protein and ligand")
        plt.ylabel("RMSD (Å)")
        plt.savefig(f"{output_directory}/RMSD_over_time.{fig_type}")

        return rmsd_df

    def rmsd_dist_frames(
        self, fig_type: str, lig: str, nucleic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the RMSD between all frames in a matrix.

        Args:
            fig_type (str): Type of the figure to save (e.g., 'png', 'jpg').
            lig (str): ligand name saved in the above pdb file. Selection string for the atomgroup to be investigated, also used during alignment.
            nucleic (bool, optional): Bool indicating if the receptor to be analyzed contains nucleic acids. Defaults to False.

        Returns:
            np.array: pairwise_rmsd_prot. Numpy array of RMSD values for pairwise protein structures.
            np.array: pairwise_rmsd_lig. Numpy array of RMSD values for ligand structures.
        """
        if nucleic:
            pairwise_rmsd_prot: np.ndarray = (
                diffusionmap.DistanceMatrix(self.universe, select="nucleic")
                .run()
                .dist_matrix
            )
        else:
            pairwise_rmsd_prot: np.ndarray = (
                diffusionmap.DistanceMatrix(self.universe, select="protein")
                .run()
                .dist_matrix
            )
        pairwise_rmsd_lig: np.ndarray = (
            diffusionmap.DistanceMatrix(self.universe, f"resname {lig}")
            .run()
            .dist_matrix
        )

        max_dist: float = max(np.amax(pairwise_rmsd_lig), np.amax(pairwise_rmsd_prot))

        fig, ax = plt.subplots(1, 2)
        fig.suptitle("RMSD between the frames")

        # protein image
        img1 = ax[0].imshow(pairwise_rmsd_prot, cmap="viridis", vmin=0, vmax=max_dist)
        if nucleic:
            ax[0].title.set_text("nucleic")
        else:
            ax[0].title.set_text("protein")
        ax[0].set_xlabel("frames")
        ax[0].set_ylabel("frames")

        # ligand image
        img2 = ax[1].imshow(pairwise_rmsd_lig, cmap="viridis", vmin=0, vmax=max_dist)
        ax[1].title.set_text("ligand")
        ax[1].set_xlabel("frames")

        fig.colorbar(
            img1, ax=ax, orientation="horizontal", fraction=0.1, label="RMSD (Å)"
        )

        plt.savefig(f"{output_directory}/RMSD_between_the_frames.{fig_type}")
        return pairwise_rmsd_prot, pairwise_rmsd_lig

    def calc_rmsd_2frames(self, ref: np.ndarray, frame: np.ndarray) -> float:
        """
        RMSD calculation between a reference and a frame.
        """
        return calc_rmsd_2frames_jit(ref, frame)

    def calculate_distance_matrix(self, selection: str) -> np.ndarray:
        distances: np.ndarray = np.zeros(
            (len(self.universe.trajectory), len(self.universe.trajectory))
        )
        # calculate distance matrix
        for i in tqdm(
            range(len(self.universe.trajectory)),
            desc="\033[1mCalculating distance matrix:\033[0m",
        ):
            self.universe.trajectory[i]
            frame_i: np.ndarray = self.universe.select_atoms(selection).positions
            for j in range(i + 1, len(self.universe.trajectory)):
                self.universe.trajectory[j]
                frame_j: np.ndarray = self.universe.select_atoms(selection).positions
                rmsd: float = self.calc_rmsd_2frames(frame_i, frame_j)
                distances[i][j] = rmsd
                distances[j][i] = rmsd
        return distances

    def calculate_representative_frame(
        self, binding_mode_frames: List[int], distance_matrix: np.ndarray
    ) -> int:
        """Calculates the most representative frame for a binding mode. This is based upon the average RMSD of a frame to all other frames in the binding mode.

        Args:
            binding_mode_frames (list): List of frames belonging to a binding mode.
            distance_matrix (np.array): Distance matrix of trajectory.

        Returns:
            int: Number of the most representative frame.
        """
        mean_rmsd_per_frame: dict = {}
        for frame_i in binding_mode_frames:
            mean_rmsd_per_frame[frame_i] = 0
            for frame_j in binding_mode_frames:
                if frame_j != frame_i:
                    mean_rmsd_per_frame[frame_i] += distance_matrix[frame_i - 1, frame_j - 1]
            mean_rmsd_per_frame[frame_i] /= len(binding_mode_frames)

        repre: int = min(mean_rmsd_per_frame, key=mean_rmsd_per_frame.get)

        return repre
