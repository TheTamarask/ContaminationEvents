import copy

import numpy as np
import wntr
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
inp_file = 'networks/Net2.inp'
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
        simulation_data = results.node['quality'].loc[HOURS_OF_SIMULATION * HOUR, :]
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


def brute_force_chemical_pollution_one_source(water_network):
    water_network_dict = wntr.network.to_dict(water_network)

    dataframe_structure = {
        'iteration': [],
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': []
    }
    pollution_results = pd.DataFrame(dataframe_structure)
    # Running simulations of pollution for all the nodes
    for i in range(len(water_network_dict['nodes'])):
        t_node_to_pollute = [water_network_dict['nodes'][i]['name']]
        analysis_results = pollution_analysis(water_network, t_node_to_pollute)
        dataframe_structure['iteration'] = i
        dataframe_structure['Node'] = t_node_to_pollute[0]
        dataframe_structure['Simulation end polluted nodes [%]'] = analysis_results['Simulation end polluted nodes [%]']
        dataframe_structure['Max polluted nodes [%]'] = analysis_results['Max polluted nodes [%]']

        pollution_results.loc[len(pollution_results)] = dataframe_structure

    pollution_results_alltime = copy.deepcopy(pollution_results)
    pollution_results_alltime = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict = {
        'Alltime best node': pollution_results_alltime.iloc[0]['Node'],
        'Alltime best pollution [%]': pollution_results_alltime.iloc[0]['Max polluted nodes [%]'],
        'Sim end best node': pollution_results.iloc[0]['Node'],
        'Sim end best pollution': pollution_results_alltime.iloc[0]['Simulation end polluted nodes [%]']
    }
    return return_dict


def pollution_analysis(water_network, nodes_to_pollute):
    # Create dictionary
    wn_dict = wntr.network.to_dict(water_network)

    # Copy Water network for later use in trace analysis
    water_network_for_trace = copy.deepcopy(water_network)

    # Poison select nodes in CHEMICAL sim network
    for element in nodes_to_pollute:
        polluted_node = None
        for node in wn_dict['nodes']:
            if node['name'] == element:
                polluted_node = node
                break
        print(polluted_node['name'])
        water_network.add_source('PollutionSource_'+polluted_node['name'], polluted_node['name'], 'SETPOINT', 2000, 'PollutionPattern')
    water_network.options.quality.parameter = 'CHEMICAL'

    # Define timesteps of the simulation for CHEMICAL ...
    water_network.options.time.start_clocktime = 0
    water_network.options.time.duration = HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = 15 * MINUTE
    water_network.options.time.quality_timestep = 15 * MINUTE
    water_network.options.time.report_timestep = 15 * MINUTE
    sim = wntr.sim.EpanetSimulator(water_network)
    # and TRACE...
    water_network_for_trace.options.time.start_clocktime = 0
    water_network_for_trace.options.time.duration = HOURS_OF_SIMULATION * HOUR
    water_network_for_trace.options.time.pattern_timestep = HOUR
    water_network_for_trace.options.time.hydraulic_timestep = 15 * MINUTE
    water_network_for_trace.options.time.quality_timestep = 15 * MINUTE
    water_network_for_trace.options.time.report_timestep = 15 * MINUTE
    # Prepare results structures
    step_results_list = list()
    df_polluted_nodes_alltime = pd.DataFrame()
    results_dict = {
        'Simulation end polluted nodes [%]': [],
        'List of polluted nodes at simulation finish': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'List of polluted nodes at max pollution time': [],
        'List of polluted nodes at any time': []
    }
    step_results_dict = {
        'Step': [],
        'Time [s]': [],
        'Nodes': [],
        'Number_of_polluted_nodes': [],
        'Node status[true/false]': [],
        'Flowrates_per_polluted_node_per_step': []
    }
    flowrates_per_polluted_node_per_step_dict = {
        'Traced_node': [],
        'Flowrates': []
    }

    # Then run simulation step by step
    sim_steps = int(HOURS_OF_SIMULATION * HOUR / (15 * MINUTE))  # hours

    for step in range(0, sim_steps + 1, 1):
        wn.options.time.duration = step * (15 * MINUTE)
        simulation_results = sim.run_sim()
        pollution_data = simulation_results.node['quality'].loc[step * (15 * MINUTE), :]

        # Get list of polluted nodes
        # Extract the number of junctions that are polluted above the threshold
        series_polluted_nodes = pollution_data.ge(POLLUTION_THRESHOLD)
        series_polluted_nodes_trimmed = series_polluted_nodes[series_polluted_nodes]
        df_polluted_nodes = series_polluted_nodes.to_frame().reset_index()
        df_polluted_nodes = df_polluted_nodes.rename(columns={step * (15 * MINUTE): 'pollution'})
        number_of_polluted_nodes = sum(series_polluted_nodes_trimmed)

        # Fill out alltime pollution DF
        #
        # TO DO
        #
        # if step == 0:
        #    df_polluted_nodes_alltime = copy.deepcopy(series_polluted_nodes)
        # else:
        #    for index in range(0, len(df_polluted_nodes)):
        #        if df_polluted_nodes.iloc[[index]]['pollution']==True:
        #            df_polluted_nodes_alltime.iloc[[index]]['pollution'] = True
        #
        # TO DO
        #
        # start trace simulation for each of polluted nodes
        list_flowrates = list()
        if number_of_polluted_nodes == 0:
            list_flowrates.append(None)
        else:
            for node_name in df_polluted_nodes['name']:
                traced_node = None
                for node in wn_dict['nodes']:
                    if node['name'] == node_name:
                        traced_node = node
                        break
                print("Symulacja kolejnego node: " + traced_node['name'])
                # Do poprawy później
                # trace_results = simulate_trace(water_network_for_trace, traced_node, step)
                # list_flowrates.append(trace_results)
        print("Przygotowanie wyników")
        step_results_dict = {
            'Step': step,
            'Time [s]': step * (15 * MINUTE),
            'Nodes': nodes_to_pollute,
            'Number_of_polluted_nodes': number_of_polluted_nodes,
            'Node status[true/false]': df_polluted_nodes,
            'Flowrates_per_polluted_node_per_step': list_flowrates
        }
        step_results_list.append(step_results_dict)
        print('fin_step: ' + str(step))

    last_step_pollution_percent = round(step_results_list[-1]['Number_of_polluted_nodes']/len(wn_dict['nodes'])*100, 2)
    max_polluted = 0
    max_polluted_step = 0
    max_polluted_time = 0
    for i in step_results_list:
        max_polluted_new = i['Number_of_polluted_nodes']
        if max_polluted_new > max_polluted:
            max_polluted = max_polluted_new
            max_polluted_step = i['Step']
            max_polluted_time = i['Time [s]']

    results_dict = {
        'Simulation end polluted nodes [%]': last_step_pollution_percent,
        'List of polluted nodes at simulation finish': step_results_list[-1]['Node status[true/false]'],
        'Max polluted nodes [%]': round(max_polluted/len(wn_dict['nodes'])*100, 2),
        'Timestamp max polluted nodes': max_polluted_time,
        'List of polluted nodes at max pollution time': step_results_list[max_polluted_step]['Node status[true/false]'],
        'List of polluted nodes at any time': step_results_list[max_polluted_step]['Node status[true/false]'] # Wrong, to correct
    }
    print('fin_final')
    return results_dict


def simulate_trace(water_network, trace_node, step):
    inj_node = trace_node['name']

    water_network.options.quality.parameter = 'TRACE'
    water_network.options.quality.trace_node = inj_node
    water_network.options.time.duration = step * (15 * MINUTE)
    # Simulate hydraulics
    sim = wntr.sim.EpanetSimulator(water_network)
    results = sim.run_sim()

    flowrates = results.node['quality'].loc[step * (15 * MINUTE), :]
    flowrates = flowrates.to_frame()
    # Prepare results
    results_dictionary = {
        'Node': [trace_node['name']],
        'Flowrates': [flowrates]
    }
    return results_dictionary


# analyze_chemical_pollution(NUMBER_OF_ITERATIONS, wn)
brute_force_chemical_pollution_one_source(wn)
# select_nodes = ["121"]
# pollution_analysis(wn, select_nodes)
# ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=trace['52200'], node_size=10,
# title='Pollution in the system')
