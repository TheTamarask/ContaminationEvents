import numpy as np
from random import random
import wntr
import wntr.network.controls as controls
import pandas as pd

# Constants
HOUR = 3600
HOURS_OF_SIMULATION = 10
START_HOUR_OF_POLLUTION = 4
END_HOUR_OF_POLLUTION = 5
NUMBER_OF_ITERATIONS = 25
POLLUTION_THRESHOLD = 300


def analyze_chemical_pollution(iterations):
    # Add source of chemical pollution to the first joint from the dictionary
    wn.options.quality.parameter = 'CHEMICAL'
    polluted_node = wn_dict['nodes'][0]

    # Create pattern for a new source of pollution
    source_pattern = wntr.network.elements.Pattern.binary_pattern('SourcePattern',
                                                                  start_time=START_HOUR_OF_POLLUTION * HOUR,
                                                                  end_time=END_HOUR_OF_POLLUTION * HOUR,
                                                                  duration=wn.options.time.duration,
                                                                  step_size=wn.options.time.pattern_timestep)
    wn.add_pattern('SourcePattern', source_pattern)

    # Running few simulations of pollution
    dataframe_structure = {
        'iteration': [],
        'Node': [],
        'Number_of_polluted_nodes': []
    }
    pollution_results = pd.DataFrame(dataframe_structure)
    i = 0
    while i < iterations:
        i += 1

        # Select node to be polluted
        polluted_node_index = np.random.randint(0, len(wn_dict['nodes']))
        polluted_node = wn_dict['nodes'][polluted_node_index]
        print(polluted_node['name'])
        # Create a new source of pollution
        wn.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'SourcePattern')

        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        # Plot results on the network
        simulation_data = results.node['quality'].loc[5 * 3600, :]
        ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=simulation_data, node_size=10, title='Pollution in the system')

        # Extract the number of junctions that are polluted above the threshold
        polluted_nodes = simulation_data.ge(POLLUTION_THRESHOLD)
        number_of_polluted_nodes = sum(polluted_nodes)
        temp_dict = {"iteration": i, "Node": polluted_node['name'],
                     "Number_of_polluted_nodes": number_of_polluted_nodes}
        pollution_results.loc[len(pollution_results)] = temp_dict
    pollution_results = pollution_results.sort_values(by=['Number_of_polluted_nodes'], ascending=False)
    print(pollution_results)


def brute_force_chemical_pollution(water_network_dict):

    wn.options.quality.parameter = 'CHEMICAL'
    # Create pattern for a new source of pollution
    source_pattern = wntr.network.elements.Pattern.binary_pattern('SourcePattern',
                                                                  start_time=START_HOUR_OF_POLLUTION * HOUR,
                                                                  end_time=END_HOUR_OF_POLLUTION * HOUR,
                                                                  duration=wn.options.time.duration,
                                                                  step_size=wn.options.time.pattern_timestep)
    wn.add_pattern('SourcePattern', source_pattern)

    #Get number of iterations
    iterations = len(water_network_dict['nodes'])
    i = 0

    dataframe_structure = {
        'iteration': [],
        'Node': [],
        'Number_of_polluted_nodes': []
    }
    pollution_results = pd.DataFrame(dataframe_structure)
    # Running simulations of pollution for all the nodes
    while i < iterations:
        # Add source of chemical pollution to joint from the dictionary
        polluted_node = water_network_dict['nodes'][i]
        print(polluted_node['name'])
        # Create a new source of pollution
        wn.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'SourcePattern')

        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        # Plot results on the network
        simulation_data = results.node['quality'].loc[5 * 3600, :]
        #ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=simulation_data, node_size=10, title='Pollution in the system')

        # Extract the number of junctions that are polluted above the threshold
        polluted_nodes = simulation_data.ge(POLLUTION_THRESHOLD)
        number_of_polluted_nodes = sum(polluted_nodes)
        temp_dict = {"iteration": i, "Node": polluted_node['name'],
                     "Number_of_polluted_nodes": number_of_polluted_nodes}
        pollution_results.loc[len(pollution_results)] = temp_dict
        i += 1
    pollution_results = pollution_results.sort_values(by=['Number_of_polluted_nodes'], ascending=False)
    print(pollution_results)


# Create a water network model and dictionary
inp_file = 'networks/Net3.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
wn_dict = wntr.network.to_dict(wn)


# Amend the dictionary
#
#
#
# Add changes from dictionary to Water Network
wn = wntr.network.from_dict(wn_dict)

scenario_names = wn.junction_name_list
sim = wntr.sim.EpanetSimulator(wn)
sim.run_sim(save_hyd=True)
wn.options.quality.parameter = 'TRACE'
inj_node = '10'
print(inj_node)
wn.options.quality.trace_node = inj_node
sim_results = sim.run_sim(use_hyd=True)
trace = sim_results.node['quality']

row = trace.loc[52200]
print(row)
#ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=trace['52200'], node_size=10, title='Pollution in the system')



# Graph the network witout any additions
wntr.graphics.plot_interactive_network(wn, title="Water network system diagram")


#analyze_chemical_pollution(NUMBER_OF_ITERATIONS)
#brute_force_chemical_pollution(wn_dict)