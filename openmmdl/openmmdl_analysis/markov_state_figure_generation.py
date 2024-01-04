import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def min_transition_calculation(min_transition):
    """
    Calculates a list based on the minimum transition time provided values and returns it in factors 1, 2, 5, 10.

    Parameters
    ----------
    min_transition : float or int
        The minimum tranisiton time input for the generation of the factors.

    Returns
    -------
    list :
        List with the minimum transition time with factors 1, 2, 5, 10.
    """
    min_transitions = [min_transition, min_transition*2, min_transition*5, min_transition*10]
    
    return min_transitions

def binding_site_markov_network(total_frames, min_transitions, combined_dict, font_size=None, size_node=None):
    """
    Generate Markov Chain plots based on transition probabilities.

    Parameters
    ----------
    total_frames : int
        The number of frames in the protein-ligand MD simulation.
    min_transitions : list of int or float
        list of transition tresholds in %. A Markov Chain plot will be generated for each of the tresholds.
    combined_dict : dict 
        A dictionary with the information of the Binding Modes and their order of appearance during the simulation for all frames.
    font_size (optional) : int
        The font size for the node labels. The default value is set to 12.
    size_node (optional) : int
        The size of the nodes in the Markov Chain plot. the default value is set to 200.

    Returns
    -------
    None
    """
    font_size = 12 if font_size is None else font_size
    size_node = 200 if size_node is None else size_node

    # Calculate the number of elements in each part
    total_length = len(combined_dict['all'])
    part_length = total_length // 3
    remaining_length = total_length % 3

    # Divide the 'all_data' into three parts
    part1_length = part_length + remaining_length
    part2_length = part_length
    part1_data = combined_dict['all'][:part1_length]
    part2_data = combined_dict['all'][part1_length:part1_length + part2_length]
    part3_data = combined_dict['all'][part1_length + part2_length:]

    # Count the occurrences of each node in each part
    part1_node_occurrences = {node: part1_data.count(node) for node in set(part1_data)}
    part2_node_occurrences = {node: part2_data.count(node) for node in set(part2_data)}
    part3_node_occurrences = {node: part3_data.count(node) for node in set(part3_data)}

    # Create the legend
    legend_labels = {
        'Blue': 'Binding mode not in top 10 occurence',
        'Green': 'Binding Mode occurence mostly in first third of frames',
        'Orange': 'Binding Mode occurence mostly in second third of frames',
        'Red': 'Binding Mode occurence mostly in third third of frames',
        'Yellow': 'Binding Mode occures throughout all trajectory equally'
    }

    legend_colors = ['skyblue', 'green', 'orange', 'red', 'yellow']

    legend_handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels.values())]  
    
    # Get the top 10 nodes with the most occurrences
    node_occurrences = {node: combined_dict['all'].count(node) for node in set(combined_dict['all'])}
    top_10_nodes = sorted(node_occurrences, key=node_occurrences.get, reverse=True)[:10]

    for min_transition_percent in min_transitions:
        min_prob = min_transition_percent / 100  # Convert percentage to probability

        # Create a directed graph
        G = nx.DiGraph()

        # Count the occurrences of each transition and self-loop
        transitions = {}
        self_loops = {}
        for i in range(len(combined_dict['all']) - 1):
            current_state = combined_dict['all'][i]
            next_state = combined_dict['all'][i + 1]

            if current_state == next_state:  # Check for self-loop
                self_loop = (current_state, next_state)
                self_loops[self_loop] = self_loops.get(self_loop, 0) + 1
            else:
                transition = (current_state, next_state)
                transitions[transition] = transitions.get(transition, 0) + 1

        # Add edges to the graph with their probabilities
        for transition, count in transitions.items():
            current_state, next_state = transition
            probability = count / len(combined_dict['all']) * 100  # Convert probability to percentage
            if probability >= min_transition_percent:
                G.add_edge(current_state, next_state, weight=probability)
                print(transition)
                print(probability)
                # Include the reverse transition with a different color
                reverse_transition = (next_state, current_state)
                print(reverse_transition)
                reverse_count = transitions.get(reverse_transition, 0)  # Use the correct count for the reverse transition
                reverse_probability = reverse_count / len(combined_dict['all']) * 100
                print(reverse_probability)
                G.add_edge(next_state, current_state, weight=reverse_probability, reverse=True)

        # Add self-loops to the graph with their probabilities
        for self_loop, count in self_loops.items():
            state = self_loop[0]
            probability = count / len(combined_dict['all']) * 100  # Convert probability to percentage
            if probability >= min_transition_percent:
                G.add_edge(state, state, weight=probability)

# Calculate transition probabilities for each direction (excluding self-loops)
        transition_probabilities_forward = {}
        transition_probabilities_backward = {}
        transition_occurrences_forward = {}
        transition_occurrences_backward = {}

        print(transitions)
        for transition, count in transitions.items():
            start_state, end_state = transition
            forward_transition = (start_state, end_state)
            backward_transition = (end_state, start_state)

            # Separate counts for forward and backward transitions
            forward_count = transitions.get(forward_transition, 0)
            backward_count = transitions.get(backward_transition, 0)

            print(forward_transition)
            print("forward")
            transition_probabilities_forward[forward_transition] = forward_count / node_occurrences[start_state] * 100
            print(forward_count)
            print("start state")
            print(start_state)
            print("end state")
            print(end_state)
            print(node_occurrences[start_state])
            print(transition_probabilities_forward[forward_transition])
            transition_occurrences_forward[forward_transition] = forward_count / len(combined_dict['all']) * 100

            print("backward")
            transition_probabilities_backward[backward_transition] = backward_count / node_occurrences[end_state] * 100
            print(backward_count)
            print(node_occurrences[start_state])
            print(transition_probabilities_backward[backward_transition])
            transition_occurrences_backward[backward_transition] = backward_count / len(combined_dict['all']) * 100

        print("probablities")
        print(transition_probabilities_forward)
        print("occurences")
        print(transition_occurrences_forward)

        # Calculate self-loop probabilities
        self_loop_probabilities = {}
        self_loop_occurences = {}
        for self_loop, count in self_loops.items():
            state = self_loop[0]
            self_loop_probabilities[state] = count / node_occurrences[state]
            self_loop_occurences[state] = count / len(combined_dict['all']) * 100

        # Generate the Markov Chain plot
        plt.figure(figsize=(30, 30))  # Increased figure size
        plt.title(f"Markov Chain Plot {min_transition_percent}% Frames Transition", fontsize=35)

        # Draw nodes and edges
        pos = nx.spring_layout(G, k=2, seed=42)  # Increased distance between nodes (k=2)
        edge_colors = []

        for u, v, data in G.edges(data=True):
            weight = data['weight']

            if u == v:  # Check if it is a self-loop
                edge_colors.append('green')  # Set green color for self-loop arrows
                width = 0.1  # Make self-loop arrows smaller
                connection_style = 'arc3,rad=-0.1'  # Make the self-loops more curved and compact
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.2, edge_color=edge_colors[-1],
                                    connectionstyle=connection_style)
            elif weight >= min_transition_percent:
                edge_colors.append('black')  # Highlight significant transitions in red

                # Check if both nodes are present before adding labels
                if G.has_node(u) and G.has_node(v):
                    width = 4.0
                    forward_label = f"{transition_occurrences_forward.get((v, u), 0):.2f}% of Frames →, {transition_probabilities_forward.get((v, u), 0):.2f}% probability"
                    backward_label = f"{transition_occurrences_backward.get((u, v), 0):.2f}% of Frames ←, {transition_probabilities_backward.get((u, v), 0):.2f}% probability"
                    edge_label = f"{forward_label}\n{backward_label}"

                    connection_style = 'arc3,rad=-0.1'
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color=edge_colors[-1],
                                        connectionstyle=connection_style)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): edge_label}, font_size=13)
            else:
                edge_colors.append('grey')  # Use black for non-significant transitions
                width = 0.5

                # Check if both nodes are present before adding labels
                if G.has_node(u) and G.has_node(v):
                    forward_label = f"{transition_occurrences_forward.get((v, u), 0):.2f}% of Frames →, {transition_probabilities_forward.get((v, u), 0):.2f}% probability"
                    backward_label = f"{transition_occurrences_backward.get((u, v), 0):.2f}% of Frames ←, {transition_probabilities_backward.get((u, v), 0):.2f}% probability"
                    edge_label = f"{forward_label}\n{backward_label}"

                    connection_style = 'arc3,rad=-0.1'
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color=edge_colors[-1],
                                        connectionstyle=connection_style)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): edge_label}, font_size=13)

        # Update the node colors based on their appearance percentages in each part
        node_colors = []
        for node in G.nodes():
            if node in top_10_nodes:
                part1_percentage = part1_node_occurrences.get(node, 0) / node_occurrences[node]
                part2_percentage = part2_node_occurrences.get(node, 0) / node_occurrences[node]
                part3_percentage = part3_node_occurrences.get(node, 0) / node_occurrences[node]

                if part1_percentage > 0.5:
                    node_colors.append('green')
                elif part2_percentage > 0.5:
                    node_colors.append('orange')
                elif part3_percentage > 0.5:
                    node_colors.append('red')
                else:
                    node_colors.append('yellow')
            else:
                node_colors.append('skyblue')

        # Draw nodes with sizes correlated to occurrences and color top 10 nodes
        node_size = [size_node * node_occurrences[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)

        # Draw node labels with occurrence percentage and self-loop probability for nodes with edges
        node_labels = {}

        for node in G.nodes():
            if G.degree(node) > 0:  # Check if the node has at least one edge
                edges_with_node = [(u, v, data['weight']) for u, v, data in G.edges(data=True) if u == node or v == node]
                relevant_edges = [edge for edge in edges_with_node if edge[2] >= min_transition_percent]

                if relevant_edges:
                    if node in top_10_nodes:
                        node_occurrence_percentage = node_occurrences[node] / len(combined_dict['all']) * 100
                        self_loop_probability = self_loop_probabilities.get(node, 0) * 100
                        self_loop_occurence = self_loop_occurences.get(node, 0)
                        node_label = f"{node}\nOccurrences: {node_occurrence_percentage:.2f}%\nSelf-Loop Probability: {self_loop_probability:.2f}% \nSelf-Loop Occurence: {self_loop_occurence:.2f}%"
                        node_labels[node] = node_label
                    else:
                        node_labels[node] = node

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, font_color='black', verticalalignment="center")


        # Add the legend to the plot
        plt.legend(handles=legend_handles, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()

        # Save the plot as a PNG file
        plot_filename = f"markov_chain_plot_{min_transition_percent}.png"
        plot_path = os.path.join("Binding_Modes_Markov_States", plot_filename)
        os.makedirs("Binding_Modes_Markov_States", exist_ok=True)  # Create the folder if it doesn't exist

        plt.savefig(plot_path, dpi=300)
        plt.clf()  # Clear the current figure for the next iteration
