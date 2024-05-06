import copy
from time import perf_counter
import numpy as np
import wntr
import pandas as pd
import statistics
import pygad

# Constants
MINUTE = 60
HOUR = 60 * MINUTE
HOURS_OF_SIMULATION = 24
POLLUTION_THRESHOLD = 300
POLLUTION_AMOUNT = 2000
MAX_THREADS = 6


def brute_force_single_source(water_network_path, start_hour_of_pollution, end_hour_of_pollution):
    # Start time measure
    start_time = perf_counter()

    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    water_network_dict = wntr.network.to_dict(water_network)

    dataframe_structure = {
        'iteration': [],
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    pollution_results = pd.DataFrame(dataframe_structure)
    # Running simulations of pollution for all the nodes
    for i in range(len(water_network_dict['nodes'])):
        t_node_to_pollute = [water_network_dict['nodes'][i]['name']]
        analysis_results = pollution_analysis(water_network, t_node_to_pollute, start_hour_of_pollution,
                                              end_hour_of_pollution)
        dataframe_structure['iteration'] = i
        dataframe_structure['Node'] = t_node_to_pollute[0]
        dataframe_structure['Simulation end polluted nodes [%]'] = analysis_results['Simulation end polluted nodes [%]']
        dataframe_structure['Max polluted nodes [%]'] = analysis_results['Max polluted nodes [%]']
        dataframe_structure['Timestamp max polluted nodes'] = analysis_results['Timestamp max polluted nodes']
        dataframe_structure['Alltime polluted nodes [%]'] = analysis_results['Alltime polluted nodes [%]']

        pollution_results.loc[len(pollution_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    pollution_results = pollution_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = pollution_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = pollution_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    pollution_results = pollution_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = pollution_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = pollution_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = pollution_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = pollution_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


def brute_force_two_source(water_network_path, start_hour_of_pollution, end_hour_of_pollution):
    # Start time measure
    start_time = perf_counter()
    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    water_network_dict = wntr.network.to_dict(water_network)

    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    pollution_results = pd.DataFrame(dataframe_structure)
    # Running simulations of pollution for all the nodes
    for i in range(len(water_network_dict['nodes'])):
        for j in range(len(water_network_dict['nodes'])):
            if not i == j:
                t_nodes_to_pollute = [water_network_dict['nodes'][i]['name'], water_network_dict['nodes'][j]['name']]
                analysis_results = pollution_analysis(water_network, t_nodes_to_pollute, start_hour_of_pollution,
                                                      end_hour_of_pollution)
                dataframe_structure['Node'] = t_nodes_to_pollute[0] + ' ' + t_nodes_to_pollute[1]
                dataframe_structure['Simulation end polluted nodes [%]'] = analysis_results[
                    'Simulation end polluted nodes [%]']
                dataframe_structure['Max polluted nodes [%]'] = analysis_results['Max polluted nodes [%]']
                dataframe_structure['Timestamp max polluted nodes'] = analysis_results['Timestamp max polluted nodes']
                dataframe_structure['Alltime polluted nodes [%]'] = analysis_results['Alltime polluted nodes [%]']

                pollution_results.loc[len(pollution_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    pollution_results = pollution_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = pollution_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = pollution_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    pollution_results = pollution_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = pollution_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = pollution_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = pollution_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = pollution_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


def pollution_analysis(water_network, nodes_to_pollute, start_hour_of_pollution, end_hour_of_pollution, timestep=900):
    # Create dictionary
    wn_dict = wntr.network.to_dict(water_network)

    # Define timesteps of the simulation for CHEMICAL ...
    water_network.options.time.start_clocktime = 0
    water_network.options.time.duration = start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = timestep
    water_network.options.time.quality_timestep = timestep
    water_network.options.time.report_timestep = timestep
    sim = wntr.sim.EpanetSimulator(water_network)

    if not 'PollutionPattern' in water_network.pattern_name_list:
        # Define pollution pattern
        pollution_pattern = wntr.network.elements.Pattern.binary_pattern('PollutionPattern',
                                                                         start_time=start_hour_of_pollution * HOUR,
                                                                         end_time=end_hour_of_pollution * HOUR,
                                                                         duration=water_network.options.time.duration,
                                                                         step_size=water_network.options.time.pattern_timestep)
        water_network.add_pattern('PollutionPattern', pollution_pattern)

    # Poison select nodes in CHEMICAL sim network
    for element in nodes_to_pollute:
        water_network.add_source('PollutionSource_' + element, element, 'SETPOINT',
                                 POLLUTION_AMOUNT, 'PollutionPattern')
    water_network.options.quality.parameter = 'CHEMICAL'

    # Prepare results structures
    step_results_list = list()
    df_polluted_nodes_alltime = pd.DataFrame()
    results_dict = {
        'Simulation end polluted nodes [%]': [],
        'List of polluted nodes at simulation finish': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'List of polluted nodes at max pollution time': [],
        'Alltime polluted nodes [%]': [],
        'List of alltime polluted nodes': []
    }
    step_results_dict = {
        'Step': [],
        'Time [s]': [],
        'Nodes': [],
        'Number_of_polluted_nodes': [],
        'Node status[true/false]': []
    }

    # Then run simulation step by step
    sim_steps = int((start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR) / timestep)  # hours

    for step in range(0, sim_steps + 1, 1):
        water_network.options.time.duration = step * timestep
        simulation_results = sim.run_sim()
        pollution_data = simulation_results.node['quality'].loc[step * timestep, :]

        # Get list of polluted nodes
        # Extract the number of junctions that are polluted above the threshold
        series_polluted_nodes = pollution_data.ge(POLLUTION_THRESHOLD)
        series_polluted_nodes_trimmed = series_polluted_nodes[series_polluted_nodes]
        df_polluted_nodes = series_polluted_nodes.to_frame().reset_index()
        df_polluted_nodes = df_polluted_nodes.rename(columns={step * timestep: 'pollution'})
        number_of_polluted_nodes = sum(series_polluted_nodes_trimmed)

        # Fill out alltime pollution DF
        if step == 0:
            df_polluted_nodes_alltime = copy.deepcopy(df_polluted_nodes)
        else:
            for index in range(0, len(df_polluted_nodes)):
                if df_polluted_nodes.iloc[[index]]['pollution'].at[index]:
                    df_polluted_nodes_alltime.at[index, 'pollution'] = True

        step_results_dict = {
            'Step': step,
            'Time [s]': step * timestep,
            'Nodes': nodes_to_pollute,
            'Number_of_polluted_nodes': number_of_polluted_nodes,
            'Node status[true/false]': df_polluted_nodes
        }
        step_results_list.append(step_results_dict)

    series_alltime_polluted_nodes = df_polluted_nodes_alltime.loc[:, 'pollution']
    series_alltime_polluted_nodes_trimmed = series_alltime_polluted_nodes[series_alltime_polluted_nodes]
    alltime_pollution_percent = round(sum(series_alltime_polluted_nodes_trimmed) / len(wn_dict['nodes']) * 100, 2)
    last_step_pollution_percent = round(step_results_list[-1]['Number_of_polluted_nodes'] / len(wn_dict['nodes']) * 100,
                                        2)
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
        'Max polluted nodes [%]': round(max_polluted / len(wn_dict['nodes']) * 100, 2),
        'Timestamp max polluted nodes': max_polluted_time,
        'List of polluted nodes at max pollution time': step_results_list[max_polluted_step]['Node status[true/false]'],
        'Alltime polluted nodes [%]': alltime_pollution_percent,
        'List of alltime polluted nodes': df_polluted_nodes_alltime
    }
    return results_dict


def visualise_pollution(water_network_path, nodes_to_pollute, start_hour_of_pollution, end_hour_of_pollution):
    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    results = pollution_analysis(water_network, nodes_to_pollute, start_hour_of_pollution, end_hour_of_pollution)
    # Max pollution visualisation
    max_pollution_df = results['List of polluted nodes at max pollution time']
    max_pollution_df_trim = max_pollution_df[max_pollution_df['pollution']]
    max_pollution_series = max_pollution_df_trim['name']
    max_pollution_list = max_pollution_series.tolist()
    ax1 = wntr.graphics.plot_interactive_network(water_network, node_attribute=max_pollution_list, node_size=10,
                                                 title='Maximum pollution reach in the system', filename='maximum.html')
    # Sim end pollution visualisation
    sim_end_pollution_df = results['List of polluted nodes at simulation finish']
    sim_end_pollution_df_trim = sim_end_pollution_df[sim_end_pollution_df['pollution']]
    sim_end_pollution_series = sim_end_pollution_df_trim['name']
    sim_end_pollution_list = sim_end_pollution_series.tolist()
    ax2 = wntr.graphics.plot_interactive_network(water_network, node_attribute=sim_end_pollution_list, node_size=10,
                                                 title='Simulation finish pollution reach in the system',
                                                 filename='sim_end.html')
    # Alltime pollution visualisation
    alltime_pollution_df = results['List of polluted nodes at any time']
    alltime_pollution_df_trim = alltime_pollution_df[sim_end_pollution_df['pollution']]
    alltime_pollution_series = alltime_pollution_df_trim['name']
    alltime_pollution_list = alltime_pollution_series.tolist()
    ax3 = wntr.graphics.plot_interactive_network(water_network, node_attribute=alltime_pollution_list, node_size=10,
                                                 title='Alltime pollution reach in the system', filename='alltime.html')


def pipe_diameter_method(water_network_path, start_hour_of_pollution, end_hour_of_pollution, two_source=False):
    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    # Create Water Network dictionary
    wn_dict = water_network.to_dict()
    # Start time counter
    start_time = perf_counter()
    # Get list of diameters
    diameters = []
    for link in wn_dict['links']:
        if link['link_type'] == 'Pump':
            break
        else:
            diameters.append(link['diameter'])
    diameters = sorted(list(set(diameters)), reverse=True)

    # Get links with 2 biggest diameters
    series_big_links = water_network.query_link_attribute(attribute='diameter', operation=np.greater,
                                                          value=diameters[2])
    # Get starting nodes of big links
    selected_nodes = []
    for link_name in series_big_links.index:
        for link in wn_dict['links']:
            if link_name == link['name']:
                selected_nodes.append(link['start_node_name'])
                break
    # Check if tow source pollution is possible
    if two_source:
        if len(selected_nodes) < 2:
            print('Less than 2 nodes selected!')
            return 'Less than 2 nodes selected!'
    # Generate results for selected nodes
    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    df_results = pd.DataFrame(dataframe_structure)
    if not two_source:
        for element in selected_nodes:
            analysis_results = pollution_analysis(water_network, [element], start_hour_of_pollution,
                                                  end_hour_of_pollution)
            dataframe_structure = {
                'Node': element,
                'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
            }
            df_results.loc[len(df_results)] = dataframe_structure
    else:
        for element1 in selected_nodes:
            for element2 in selected_nodes:
                if not element1 == element2:
                    analysis_results = pollution_analysis(water_network, [element1, element2], start_hour_of_pollution,
                                                          end_hour_of_pollution)
                    dataframe_structure = {
                        'Node': str(element1 + ', ' + element2),
                        'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                        'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                        'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                        'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
                    }
                    df_results.loc[len(df_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    df_results = df_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = df_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = df_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    df_results = df_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = df_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = df_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    df_results = df_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = df_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = df_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


def select_flowrate(dict_tuple):
    return dict_tuple[1]['Flowrate']


def max_outflow_method(water_network_path, start_hour_of_pollution, end_hour_of_pollution, two_source=False):
    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    # Create Water Network dictionary
    wn_dict = water_network.to_dict()
    # Start time counter
    start_time = perf_counter()

    # Create a mapping of output links to nodes
    node_link_map = {}
    for n in wn_dict['nodes']:
        list_of_links = water_network.get_links_for_node(node_name=n['name'], flag='OUTLET')
        dictionary_row = {
            'Node': n['name'],
            'Links': list_of_links,
            "Number of links": len(list_of_links),
            'Flowrate': 0.0
        }
        node_link_map[n['name']] = dictionary_row

    # Get flowrates for simulation period

    # Define timesteps of the simulation for Flowrates ...
    water_network.options.time.start_clocktime = 0
    water_network.options.time.duration = start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = 15 * MINUTE
    water_network.options.time.quality_timestep = 15 * MINUTE
    water_network.options.time.report_timestep = 15 * MINUTE
    sim = wntr.sim.EpanetSimulator(water_network)
    results = sim.run_sim()

    for element in node_link_map.items():
        summed_flowrates = 0.0
        for link in element[1]['Links']:
            series_flowrates = results.link['flowrate']
            series_flowrates = series_flowrates.loc[:, link]
            series_flowrates = abs(series_flowrates)
            summed_flowrates += sum(series_flowrates)
        element[1]['Flowrate'] = summed_flowrates

    # Select top 10% of nodes to try for pollution
    selected_nodes = []
    n = round(0.1 * len(wn_dict['nodes']))
    if n < 2:
        n = 2
    # Sort by flowrate descending
    node_link_map = sorted(node_link_map.items(), key=select_flowrate, reverse=True)
    for i in range(n):
        selected_nodes.append(node_link_map[i][0])

    # Generate results for select nodes
    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    df_results = pd.DataFrame(dataframe_structure)
    if not two_source:
        for element in selected_nodes:
            analysis_results = pollution_analysis(water_network, [element], start_hour_of_pollution,
                                                  end_hour_of_pollution)
            dataframe_structure = {
                'Node': element,
                'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
            }
            df_results.loc[len(df_results)] = dataframe_structure
    else:
        for element1 in selected_nodes:
            for element2 in selected_nodes:
                if not element1 == element2:
                    analysis_results = pollution_analysis(water_network, [element1, element2], start_hour_of_pollution,
                                                          end_hour_of_pollution)
                    dataframe_structure = {
                        'Node': str(element1 + ', ' + element2),
                        'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                        'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                        'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                        'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
                    }
                    df_results.loc[len(df_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    df_results = df_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = df_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = df_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    df_results = df_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = df_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = df_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    df_results = df_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = df_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = df_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


def combined_method(water_network_path, start_hour_of_pollution, end_hour_of_pollution, two_source=False):
    # Create a water network model
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    # Create Water Network dictionary
    wn_dict = water_network.to_dict()
    # Start time counter
    start_time = perf_counter()

    # Select nodes by outflow
    # Create a mapping of output links to nodes
    node_link_map = {}
    for n in wn_dict['nodes']:
        list_of_links = water_network.get_links_for_node(node_name=n['name'], flag='OUTLET')
        dictionary_row = {
            'Node': n['name'],
            'Links': list_of_links,
            "Number of links": len(list_of_links),
            'Flowrate': 0.0
        }
        node_link_map[n['name']] = dictionary_row

    # Get flowrates for simulation period

    # Define timesteps of the simulation for Flowrates ...
    water_network.options.time.start_clocktime = 0
    water_network.options.time.duration = start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = 15 * MINUTE
    water_network.options.time.quality_timestep = 15 * MINUTE
    water_network.options.time.report_timestep = 15 * MINUTE
    sim = wntr.sim.EpanetSimulator(water_network)
    results = sim.run_sim()

    for element in node_link_map.items():
        summed_flowrates = 0.0
        for link in element[1]['Links']:
            series_flowrates = results.link['flowrate']
            series_flowrates = series_flowrates.loc[:, link]
            series_flowrates = abs(series_flowrates)
            summed_flowrates += sum(series_flowrates)
        element[1]['Flowrate'] = summed_flowrates

    # Select top 20% of nodes to try for pollution
    selected_nodes_outflow = []
    n = round(0.2 * len(wn_dict['nodes']))
    if n < 2:
        n = 2
    # Sort by flowrate descending
    node_link_map = sorted(node_link_map.items(), key=select_flowrate, reverse=True)
    for i in range(n):
        selected_nodes_outflow.append(node_link_map[i][0])

    # Select nodes by pipe diameter
    # Get list of diameters
    diameters = []
    for link in wn_dict['links']:
        if link['link_type'] == 'Pump':
            break
        else:
            diameters.append(link['diameter'])
    diameters = sorted(list(set(diameters)), reverse=True)

    # Get links with 2 biggest diameters
    series_big_links = water_network.query_link_attribute(attribute='diameter', operation=np.greater,
                                                          value=diameters[2])
    # Get starting nodes of big links
    selected_nodes_diameter = []
    for link_name in series_big_links.index:
        for link in wn_dict['links']:
            if link_name == link['name']:
                selected_nodes_diameter.append(link['start_node_name'])
                break

    # Compare the lists
    selected_nodes = set(selected_nodes_diameter).intersection(selected_nodes_outflow)
    # If less than 2 matches found - no two source
    if len(selected_nodes) < 2:
        if len(selected_nodes) == 0:
            print("No matches found!")
            return "No matches found!"
        if two_source:
            print('Less than 2 nodes suitable!')
            return 'Less than 2 nodes suitable!'

    # Generate results for select nodes
    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    df_results = pd.DataFrame(dataframe_structure)
    if not two_source:
        for element in selected_nodes:
            analysis_results = pollution_analysis(water_network, [element], start_hour_of_pollution,
                                                  end_hour_of_pollution)
            dataframe_structure = {
                'Node': element,
                'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
            }
            df_results.loc[len(df_results)] = dataframe_structure
    else:
        for element1 in selected_nodes:
            for element2 in selected_nodes:
                if not element1 == element2:
                    analysis_results = pollution_analysis(water_network, [element1, element2], start_hour_of_pollution,
                                                          end_hour_of_pollution)
                    dataframe_structure = {
                        'Node': str(element1 + ', ' + element2),
                        'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                        'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                        'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                        'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
                    }
                    df_results.loc[len(df_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    df_results = df_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = df_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = df_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    df_results = df_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = df_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = df_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    df_results = df_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = df_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = df_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


# objective function
def genetic_algorithm_objective(pop, water_network_path, start_hour_of_pollution, end_hour_of_pollution, mode='global',
                                two_source=False):
    # Create a water network model
    wn = wntr.network.WaterNetworkModel(water_network_path)
    wn_dict = wn.to_dict()
    # Generate results for the selected populations, use lower resolution - timestep = 1 hour
    # Store results in a list
    results = []
    if not two_source:
        for node in range(len(pop)):
            if pop[node]:
                nodes_to_pollute = [wn_dict['nodes'][node]['name']]
                result = pollution_analysis(wn, nodes_to_pollute, start_hour_of_pollution=start_hour_of_pollution,
                                            end_hour_of_pollution=end_hour_of_pollution, timestep=HOUR)
                results.append(result)
    else:
        nodes_to_pollute = []
        for node in range(len(pop)):
            if pop[node]:
                nodes_to_pollute.append(wn_dict['nodes'][node]['name'])
            if len(nodes_to_pollute) == 2:
                result = pollution_analysis(wn, nodes_to_pollute, start_hour_of_pollution=start_hour_of_pollution,
                                            end_hour_of_pollution=end_hour_of_pollution, timestep=HOUR)
                nodes_to_pollute = []
                results.append(result)
    if len(results) == 0:
        empty_result = {
            'Simulation end polluted nodes [%]': 0,
            'Max polluted nodes [%]': 0,
            'Alltime polluted nodes [%]': 0,
        }
        results.append(empty_result)
    if mode == 'max':
        temp_results = []
        for result in results:
            temp_results.append(result['Max polluted nodes [%]'])
        return statistics.median(temp_results)
    elif mode == 'sim_end':
        temp_results = []
        for result in results:
            temp_results.append(result['Simulation end polluted nodes [%]'])
        return statistics.median(temp_results)
    else:
        temp_results = []
        for result in results:
            temp_results.append(result['Alltime polluted nodes [%]'])
        return statistics.median(temp_results)


# Selection function for genetic algorithm - tournament selection used
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# Crossover function for genetic algorithm -  two parents create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# Mutation function for genetic algorithm
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# Genetic algorithm function
def genetic_algorithm(objective, n_nodes, n_iter, n_pop, r_cross, r_mut, n_nodes_in_pop, water_network_path,
                      start_hour_of_pollution, end_hour_of_pollution, mode="global", two_source=False):
    # initial population of random bitstring
    pop = []
    for pop_number in range(1, n_pop + 1):
        temp_pop = []
        for node_number in range(n_nodes):
            lower_bound = pop_number * n_nodes_in_pop - n_nodes_in_pop
            upper_bound = pop_number * n_nodes_in_pop
            if lower_bound <= node_number < upper_bound:
                temp_pop.append(1)
            else:
                temp_pop.append(0)
        pop.append(temp_pop)

    # keep track of best solution
    best, best_eval = pop[0], objective(pop[0], water_network_path, start_hour_of_pollution, end_hour_of_pollution,
                                        mode, two_source)
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c, water_network_path, start_hour_of_pollution, end_hour_of_pollution, mode, two_source)
                  for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            if i + 2 <= n_pop:
                p1, p2 = selected[i], selected[i + 1]
            else:
                p1, p2 = selected[i - 1], selected[i]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        n_pop = len(pop)
    return [best, best_eval]


def genetic_algorithm_method(water_network_path, start_hour_of_pollution, end_hour_of_pollution, n_iter, n_nodes_in_pop,
                             r_cross, r_mut, two_source=False):
    # Start time counter
    start_time = perf_counter()
    # Create water network
    water_network = wntr.network.WaterNetworkModel(water_network_path)
    wn_dict = water_network.to_dict()
    # Prepare and run the algorithm
    n_nodes = len(wn_dict['nodes'])
    n_pop = round(n_nodes / n_nodes_in_pop)
    [best_nodes, best_eval] = genetic_algorithm(genetic_algorithm_objective, n_nodes=n_nodes, n_iter=n_iter,
                                                n_pop=n_pop,
                                                r_cross=r_cross, r_mut=r_mut, n_nodes_in_pop=n_nodes_in_pop,
                                                water_network_path="networks/Net1.inp", start_hour_of_pollution=5,
                                                end_hour_of_pollution=7, mode='global', two_source=two_source)
    # Prepare selected nodes
    selected_nodes = []
    for node_nr in range(len(wn_dict['nodes'])):
        if best_nodes[node_nr]:
            temp = wn_dict['nodes'][node_nr]['name']
            selected_nodes.append(temp)
    # Generate results for select nodes
    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': [],
        'Timestamp max polluted nodes': [],
        'Alltime polluted nodes [%]': []
    }
    df_results = pd.DataFrame(dataframe_structure)
    if not two_source:
        for element in selected_nodes:
            analysis_results = pollution_analysis(water_network, [element], start_hour_of_pollution,
                                                  end_hour_of_pollution)
            dataframe_structure = {
                'Node': element,
                'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
            }
            df_results.loc[len(df_results)] = dataframe_structure
    else:
        for element1 in selected_nodes:
            for element2 in selected_nodes:
                if not element1 == element2:
                    analysis_results = pollution_analysis(water_network, [element1, element2], start_hour_of_pollution,
                                                          end_hour_of_pollution)
                    dataframe_structure = {
                        'Node': str(element1 + ', ' + element2),
                        'Simulation end polluted nodes [%]': analysis_results['Simulation end polluted nodes [%]'],
                        'Max polluted nodes [%]': analysis_results['Max polluted nodes [%]'],
                        'Timestamp max polluted nodes': analysis_results['Timestamp max polluted nodes'],
                        'Alltime polluted nodes [%]': analysis_results['Alltime polluted nodes [%]']
                    }
                    df_results.loc[len(df_results)] = dataframe_structure

    # Prepare results
    return_dict = {
        'Alltime best node': [],
        'Alltime best pollution [%]': [],
        'Max polluted source node': [],
        'Max polluted nodes [%]': [],
        'Sim end best node': [],
        'Sim end best pollution': [],
        'Duration': []
    }
    # Alltime
    df_results = df_results.sort_values(by=['Alltime polluted nodes [%]'], ascending=False)
    return_dict['Alltime best node'] = df_results.iloc[0]['Node']
    return_dict['Alltime best pollution [%]'] = df_results.iloc[0]['Alltime polluted nodes [%]']
    # Max polluted
    df_results = df_results.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    return_dict['Max polluted source node'] = df_results.iloc[0]['Node'],
    return_dict['Max polluted nodes [%]'] = df_results.iloc[0]['Max polluted nodes [%]'],
    # Sim end
    df_results = df_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    return_dict['Sim end best node'] = df_results.iloc[0]['Node']
    return_dict['Sim end best pollution'] = df_results.iloc[0]['Simulation end polluted nodes [%]']
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time
    return_dict['Duration'] = duration

    return return_dict


def get_results_bruteforce(water_network_path, two_source=False):
    # Get the results form first window
    if not two_source:
        results_dict = brute_force_single_source(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7)
        filename = "results/brute_force_single_results_5-7_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7)
        filename = "results/brute_force_double_results_5-7_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 5-7 done!')

    # Get the results form second window
    if not two_source:
        results_dict = brute_force_single_source(water_network_path, start_hour_of_pollution=13,
                                                 end_hour_of_pollution=15)
        filename = "results/brute_force_single_results_13-15_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15)
        filename = "results/brute_force_double_results_13-15_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 13-15 done!')

    # Get the results form third window
    if not two_source:
        results_dict = brute_force_single_source(water_network_path, start_hour_of_pollution=18,
                                                 end_hour_of_pollution=20)
        filename = "results/brute_force_single_results_18-20_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20)
        filename = "results/brute_force_double_results_18-20_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 18-20 done!')


def get_results_pipe_diameter(water_network_path, two_source=False):
    # Get the results form first window
    if not two_source:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                            two_source=False)
        filename = "results/pipe_diameter_single_results_5-7_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                            two_source=True)
        filename = "results/pipe_diameter_double_results_5-7_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Pipe diameter 5-7 done!')

    # Get the results form second window

    if not two_source:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                            two_source=False)
        filename = "results/pipe_diameter_single_results_13-15_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                            two_source=True)
        filename = "results/pipe_diameter_double_results_13-15_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Pipe diameter 13-15 done!')

    # Get the results form third window

    if not two_source:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                            two_source=False)
        filename = "results/pipe_diameter_single_results_18-20_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = pipe_diameter_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                            two_source=True)
        filename = "results/pipe_diameter_double_results_18-20_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Pipe diameter 18-20 done!')


def get_results_max_outflow(water_network_path, two_source=False):
    # Get the results form first window
    if not two_source:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                          two_source=False)
        filename = "results/max_outflow_method_single_results_5-7_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                          two_source=True)
        filename = "results/max_outflow_method_double_results_5-7_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Max outflow 5-7 done!')

    # Get the results form second window
    if not two_source:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                          two_source=False)
        filename = "results/max_outflow_single_results_13-15_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                          two_source=True)
        filename = "results/max_outflow_double_results_13-15_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Max outflow 13-15 done!')

    # Get the results form third window
    if not two_source:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                          two_source=False)
        filename = "results/max_outflow_single_results_18-20_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = max_outflow_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                          two_source=True)
        filename = "results/max_outflow_double_results_18-20_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Max outflow 18-20 done!')


def get_results_combined(water_network_path, two_source=False):
    # Get the results form first window
    if not two_source:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                       two_source=False)
        filename = "results/max_outflow_method_single_results_5-7_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                       two_source=True)
        filename = "results/combined_method_double_results_5-7_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Combined 5-7 done!')

    # Get the results form second window
    if not two_source:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                       two_source=False)
        filename = "results/combined_method_single_results_13-15_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=13, end_hour_of_pollution=15,
                                       two_source=True)
        filename = "results/combined_method_double_results_13-15_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Combined 13-15 done!')

    # Get the results form third window
    if not two_source:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                       two_source=False)
        filename = "results/combined_method_single_results_18-20_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = combined_method(water_network_path, start_hour_of_pollution=18, end_hour_of_pollution=20,
                                       two_source=True)
        filename = "results/combined_method_double_results_18-20_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Combined 18-20 done!')


def get_results_genetic(water_network_path, n_iter, n_nodes_in_pop, r_cross, r_mut, two_source=False):
    # Get the results form first window
    if not two_source:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=False)
        filename = "results/genetic_algorithm_method_single_results_5-7_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=5, end_hour_of_pollution=7,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=True)
        filename = "results/genetic_algorithm_method_double_results_5-7_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Genetic algorithm 5-7 done!')

    # Get the results form second window
    if not two_source:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=13,
                                                end_hour_of_pollution=15,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=False)
        filename = "results/genetic_algorithm_method_single_results_13-15_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=13,
                                                end_hour_of_pollution=15,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=True)
        filename = "results/genetic_algorithm_method_double_results_13-15_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Genetic algorithm 13-15 done!')

    # Get the results form third window
    if not two_source:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=18,
                                                end_hour_of_pollution=20,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=False)
        filename = "results/genetic_algorithm_method_single_results_18-20_window_" + water_network_path[9:] + ".txt"
    else:
        results_dict = genetic_algorithm_method(water_network_path, start_hour_of_pollution=18,
                                                end_hour_of_pollution=20,
                                                n_iter=n_iter, n_nodes_in_pop=n_nodes_in_pop, r_cross=r_cross,
                                                r_mut=r_mut, two_source=True)
        filename = "results/genetic_algorithm_method_double_results_18-20_window_" + water_network_path[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Genetic algorithm 18-20 done!')


if __name__ == '__main__':
    # Pipe diameter
    get_results_pipe_diameter("networks/Net1.inp")
    get_results_pipe_diameter("networks/Net1.inp", two_source=True)
    print('Net1 pipe diameter done!')
    get_results_pipe_diameter("networks/Net3.inp")
    get_results_pipe_diameter("networks/Net3.inp", two_source=True)
    print('Net3 pipe diameter done!')
    get_results_pipe_diameter("networks/LongTermImprovement.inp")
    get_results_pipe_diameter("networks/LongTermImprovement.inp", two_source=True)
    print('Long Term Improvement pipe diameter done!')

    # Max outflow
    get_results_max_outflow("networks/Net1.inp")
    get_results_max_outflow("networks/Net1.inp", two_source=True)
    print('Net1 max outflow done!')
    get_results_max_outflow("networks/Net3.inp")
    get_results_max_outflow("networks/Net3.inp", two_source=True)
    print('Net3 max outflow done!')
    get_results_max_outflow("networks/LongTermImprovement.inp")
    get_results_max_outflow("networks/LongTermImprovement.inp", two_source=True)
    print('Long Term Improvement max outflow done!')

    # Combined
    get_results_combined("networks/Net1.inp")
    get_results_combined("networks/Net1.inp", two_source=True)
    print('Net1 combined done!')
    get_results_combined("networks/Net3.inp")
    get_results_combined("networks/Net3.inp", two_source=True)
    print('Net3 combined done!')
    get_results_combined("networks/LongTermImprovement.inp")
    get_results_combined("networks/LongTermImprovement.inp", two_source=True)
    print('Long Term Improvement combined done!')

    # Genetic
    get_results_genetic("networks/Net1.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1)
    get_results_genetic("networks/Net1.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1,
                        two_source=True)
    print('Net1 genetic done!')
    get_results_genetic("networks/Net3.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1)
    get_results_genetic("networks/Net3.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1,
                        two_source=True)
    print('Net3 genetic done!')
    get_results_genetic("networks/LongTermImprovement.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1)
    get_results_genetic("networks/LongTermImprovement.inp", n_iter=10, n_nodes_in_pop=4, r_cross=0.9, r_mut=0.1,
                        two_source=True)
    print('Long Term Improvement genetic done!')

    # Single source brute force
    get_results_bruteforce("networks/Net1.inp")
    print('Net1 brute force done!')
    get_results_bruteforce("networks/Net3.inp")
    print('Net3 brute force done!')
    get_results_bruteforce("networks/LongTermImprovement.inp")
    print('LTI brute force done!')

    # Dual source brute force
    get_results_bruteforce(water_network_path="networks/Net1.inp", two_source=True)
    print('Net1 brute force done!')
    get_results_bruteforce(water_network_path="networks/Net3.inp", two_source=True)
    print('Net3 brute force done!')

    # get_results_bruteforce(water_network_path="networks/LongTermImprovement.inp", two_source=True)
    # print('LTI brute force done!'
