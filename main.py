import copy

import numpy as np
from random import random
import wntr
import wntr.network.controls as controls
import pandas as pd

# Constants
MINUTE = 60
HOUR = 60 * MINUTE
HOURS_OF_SIMULATION = 24
START_HOUR_OF_POLLUTION = 0
END_HOUR_OF_POLLUTION = 5
NUMBER_OF_ITERATIONS = 5
POLLUTION_THRESHOLD = 300

# Create a water network model and dictionary
inp_file = 'networks/Net3.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

# Define pollution pattern
pollution_pattern = wntr.network.elements.Pattern.binary_pattern('PollutionPattern',
                                                                 start_time=START_HOUR_OF_POLLUTION * HOUR,
                                                                 end_time=END_HOUR_OF_POLLUTION * HOUR,
                                                                 duration=wn.options.time.duration,
                                                                 step_size=wn.options.time.pattern_timestep)
wn.add_pattern('PollutionPattern', pollution_pattern)


def analyze_chemical_pollution(iterations, water_network):
    wn_dict = wntr.network.to_dict(water_network)
    water_network.options.quality.parameter = 'CHEMICAL'

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
        water_network.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 1000, 'PollutionPattern')

        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(water_network)
        results = sim.run_sim()

        # Plot results on the network
        simulation_data = results.node['quality'].loc[5 * 3600, :]
        ax1 = wntr.graphics.plot_interactive_network(water_network, node_attribute=simulation_data, node_size=10,
                                                     title='Pollution in the system')

        # Extract the number of junctions that are polluted above the threshold
        polluted_nodes = simulation_data.ge(POLLUTION_THRESHOLD)
        number_of_polluted_nodes = sum(polluted_nodes)
        temp_dict = {"iteration": i, "Node": polluted_node['name'],
                     "Number_of_polluted_nodes": number_of_polluted_nodes}
        pollution_results.loc[len(pollution_results)] = temp_dict
    pollution_results = pollution_results.sort_values(by=['Number_of_polluted_nodes'], ascending=False)
    print(pollution_results)


def brute_force_chemical_pollution(water_network):
    water_network_dict = wntr.network.to_dict(water_network)
    water_network.options.quality.parameter = 'CHEMICAL'

    # Get number of iterations
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
        water_network.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'PollutionPattern')

        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(water_network)
        results = sim.run_sim()

        # Plot results on the network
        simulation_data = results.node['quality'].loc[5 * 3600, :]
        # ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=simulation_data, node_size=10, title='Pollution in the system')

        # Extract the number of junctions that are polluted above the threshold
        polluted_nodes = simulation_data.ge(POLLUTION_THRESHOLD)
        number_of_polluted_nodes = sum(polluted_nodes)
        temp_dict = {"iteration": i, "Node": polluted_node['name'],
                     "Number_of_polluted_nodes": number_of_polluted_nodes}
        pollution_results.loc[len(pollution_results)] = temp_dict
        i += 1
    pollution_results = pollution_results.sort_values(by=['Number_of_polluted_nodes'], ascending=False)
    print(pollution_results)


def pollution_analysis(water_network):
    # Create dictionary
    wn_dict = wntr.network.to_dict(water_network)

    # Copy Water network for later use in trace analysis
    water_network_for_trace = copy.deepcopy(water_network)

    # Poison select node in CHEMICAL sim network
    # TEMP: select random node
    polluted_node_index = np.random.randint(0, len(wn_dict['nodes']))
    polluted_node = wn_dict['nodes'][polluted_node_index]
    print(polluted_node['name'])
    water_network.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'PollutionPattern')
    water_network.options.quality.parameter = 'CHEMICAL'

    # Define timesteps of the simulation for CHEMICAL ...
    water_network.options.time.duration = HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = 15 * MINUTE
    water_network.options.time.quality_timestep = 15 * MINUTE
    sim = wntr.sim.EpanetSimulator(water_network)
    # and TRACE...
    water_network_for_trace.options.time.duration = HOURS_OF_SIMULATION * HOUR
    water_network_for_trace.options.time.pattern_timestep = HOUR
    water_network_for_trace.options.time.hydraulic_timestep = 15 * MINUTE
    water_network_for_trace.options.time.quality_timestep = 15 * MINUTE
    # prepare master hyd file
    sim.run_sim(save_hyd=True, file_prefix='master')

    # water_network.options.quality.parameter = 'TRACE'

    # inj_node = polluted_node['name']
    # print(inj_node)
    # water_network.options.quality.trace_node = inj_node

    # Prepare results DF
    dataframe_structure = {
        'Time [s]': [],
        'Node': [],
        'Number_of_polluted_nodes': []
    }
    polluted_nodes_results = pd.DataFrame(dataframe_structure)

    # Then run simulation step by step

    sim_steps = int(HOURS_OF_SIMULATION * HOUR / (15 * MINUTE))  # hours

    for step in range(0, sim_steps + 1, 1):
        wn.options.time.duration = step * (15 * MINUTE)
        simulation_results = sim.run_sim()
        pollution_data = simulation_results.node['quality'].loc[step * (15 * MINUTE), :]

        # Get list of polluted nodes
        # Extract the number of junctions that are polluted above the threshold
        list_polluted_nodes = pollution_data.ge(POLLUTION_THRESHOLD)
        list_polluted_nodes = list_polluted_nodes.to_frame().reset_index()
        list_polluted_nodes = list_polluted_nodes.rename(columns={step * (15 * MINUTE): 'pollution'})
        list_polluted_nodes = list_polluted_nodes[list_polluted_nodes.pollution]
        number_of_polluted_nodes = sum(list_polluted_nodes.pollution)
        temp_dict = {"Time [s]": step * (15 * MINUTE), "Node": polluted_node['name'],
                     "Number_of_polluted_nodes": number_of_polluted_nodes}
        polluted_nodes_results.loc[len(polluted_nodes_results)] = temp_dict

        # start trace simulation for each of polluted nodes
        for node_name in list_polluted_nodes['name']:
            traced_node = None
            for node in wn_dict['nodes']:
                if node['name'] == node_name:
                    traced_node = node
                    break
            print("Symulacja kolejnego node: " + traced_node['name'])
            simulate_trace(water_network_for_trace, traced_node)

    print('fin')


def simulate_trace(water_network, trace_node):
    water_network.options.quality.parameter = 'TRACE'
    print('Symulacja rozpływu node: ', trace_node['name'])
    water_network.options.quality.trace_node = trace_node
    sim = wntr.sim.EpanetSimulator(water_network)


def old_pollution_analysis(water_network):
    scenario_names = water_network.junction_name_list
    sim = wntr.sim.EpanetSimulator(water_network)
    sim.run_sim(save_hyd=True, file_prefix='master')
    water_network.options.quality.parameter = 'TRACE'
    inj_node = '10'
    print(inj_node)
    water_network.options.quality.trace_node = inj_node
    sim_results = sim.run_sim(use_hyd=True)
    trace = sim_results.node['quality']
    row = trace.loc[52200]
    print(row)

    # Graph the network witout any additions
    wntr.graphics.plot_interactive_network(water_network, title="Water network system diagram")
    # ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=trace['52200'], node_size=10, title='Pollution in the system')


# analyze_chemical_pollution(NUMBER_OF_ITERATIONS, wn)
# brute_force_chemical_pollution(wn)
pollution_analysis(wn)
# ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=trace['52200'], node_size=10, title='Pollution in the system')
