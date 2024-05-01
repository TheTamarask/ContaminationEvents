import copy
from time import perf_counter
import numpy as np
import wntr
import pandas as pd
import concurrent.futures

# Constants
MINUTE = 60
HOUR = 60 * MINUTE
HOURS_OF_SIMULATION = 24
POLLUTION_THRESHOLD = 300
POLLUTION_AMOUNT = 2000
MAX_THREADS = 6


def brute_force_single_source(water_network, start_hour_of_pollution, end_hour_of_pollution):
    # Start time measure
    start_time = perf_counter()
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
    # Alltime
    pollution_results_alltime = copy.deepcopy(pollution_results)
    pollution_results_alltime = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    # Max polluted
    pollution_results_max_polluted = copy.deepcopy(pollution_results)
    pollution_results_max_polluted = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'],
                                                                           ascending=False)
    # Sim end
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time

    return_dict = {
        'Alltime best node': pollution_results_alltime.iloc[0]['Node'],
        'Alltime best pollution [%]': pollution_results_alltime.iloc[0]['Max polluted nodes [%]'],
        'Max polluted source node': pollution_results_max_polluted.iloc[0]['Node'],
        'Max polluted nodes [%]': pollution_results_max_polluted.iloc[0]['Max polluted nodes [%]'],
        'Sim end best node': pollution_results.iloc[0]['Node'],
        'Sim end best pollution': pollution_results_alltime.iloc[0]['Simulation end polluted nodes [%]'],
        'Duration': duration
    }
    return return_dict


def brute_force_two_source(water_network, start_hour_of_pollution, end_hour_of_pollution):
    # Start time measure
    start_time = perf_counter()
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
    # Alltime
    pollution_results_alltime = copy.deepcopy(pollution_results)
    pollution_results_alltime = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    # Max polluted
    pollution_results_max_polluted = copy.deepcopy(pollution_results)
    pollution_results_max_polluted = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'],
                                                                           ascending=False)
    # Sim end
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time

    return_dict = {
        'Alltime best node': pollution_results_alltime.iloc[0]['Node'],
        'Alltime best pollution [%]': pollution_results_alltime.iloc[0]['Max polluted nodes [%]'],
        'Max polluted source node': pollution_results_max_polluted.iloc[0]['Node'],
        'Max polluted nodes [%]': pollution_results_max_polluted.iloc[0]['Max polluted nodes [%]'],
        'Sim end best node': pollution_results.iloc[0]['Node'],
        'Sim end best pollution': pollution_results_alltime.iloc[0]['Simulation end polluted nodes [%]'],
        'Duration': duration
    }
    return return_dict


def pollution_analysis(water_network, nodes_to_pollute, start_hour_of_pollution, end_hour_of_pollution):
    # Create dictionary
    wn_dict = wntr.network.to_dict(water_network)

    # Define timesteps of the simulation for CHEMICAL ...
    water_network.options.time.start_clocktime = 0
    water_network.options.time.duration = start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR
    water_network.options.time.pattern_timestep = HOUR
    water_network.options.time.hydraulic_timestep = 15 * MINUTE
    water_network.options.time.quality_timestep = 15 * MINUTE
    water_network.options.time.report_timestep = 15 * MINUTE
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
        polluted_node = None
        for node in wn_dict['nodes']:
            if node['name'] == element:
                polluted_node = node
                break
        water_network.add_source('PollutionSource_' + polluted_node['name'], polluted_node['name'], 'SETPOINT',
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
        'List of polluted nodes at any time': []
    }
    step_results_dict = {
        'Step': [],
        'Time [s]': [],
        'Nodes': [],
        'Number_of_polluted_nodes': [],
        'Node status[true/false]': []
    }
    flowrates_per_polluted_node_per_step_dict = {
        'Traced_node': [],
        'Flowrates': []
    }

    # Then run simulation step by step
    sim_steps = int((start_hour_of_pollution * HOUR + HOURS_OF_SIMULATION * HOUR) / (15 * MINUTE))  # hours

    for step in range(0, sim_steps + 1, 1):
        water_network.options.time.duration = step * (15 * MINUTE)
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
        if step == 0:
            df_polluted_nodes_alltime = copy.deepcopy(df_polluted_nodes)
        else:
            for index in range(0, len(df_polluted_nodes)):
                if df_polluted_nodes.iloc[[index]]['pollution'].at[index]:
                    test = df_polluted_nodes_alltime.at[index, 'pollution']
                    df_polluted_nodes_alltime.at[index, 'pollution'] = True
                    test = df_polluted_nodes_alltime.at[index, 'pollution']

        step_results_dict = {
            'Step': step,
            'Time [s]': step * (15 * MINUTE),
            'Nodes': nodes_to_pollute,
            'Number_of_polluted_nodes': number_of_polluted_nodes,
            'Node status[true/false]': df_polluted_nodes
        }
        step_results_list.append(step_results_dict)

    series_alltime_polluted_nodes = df_polluted_nodes_alltime.loc[:, 'pollution']
    series_alltime_polluted_nodes_trimmed = series_alltime_polluted_nodes[series_alltime_polluted_nodes]
    alltime_pollution_percent = round(sum(series_alltime_polluted_nodes_trimmed) / len(wn_dict['nodes']), 2)

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


def visualise_pollution(water_network, nodes_to_pollute, start_hour_of_pollution, end_hour_of_pollution):
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


def get_results_bruteforce(water_network_path, two_source=False):
    # Create a water network model
    inp_file = water_network_path
    wn = wntr.network.WaterNetworkModel(inp_file)
    # Get the results form first window
    if not two_source:
        results_dict = brute_force_single_source(wn, start_hour_of_pollution=5, end_hour_of_pollution=7)
        filename = "brute_force_single_results_5-7_window_" + inp_file[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(wn, start_hour_of_pollution=5, end_hour_of_pollution=7)
        filename = "brute_force_double_results_5-7_window_" + inp_file[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 5-7 done!')

    # Get the results form second window
    # Create a water network model
    inp_file = water_network_path
    wn = wntr.network.WaterNetworkModel(inp_file)
    if not two_source:
        results_dict = brute_force_single_source(wn, start_hour_of_pollution=13, end_hour_of_pollution=15)
        filename = "brute_force_single_results_13-15_window_" + inp_file[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(wn, start_hour_of_pollution=13, end_hour_of_pollution=15)
        filename = "brute_force_double_results_13-15_window_" + inp_file[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 13-15 done!')

    # Get the results form third window
    # Create a water network model
    inp_file = water_network_path
    wn = wntr.network.WaterNetworkModel(inp_file)
    if not two_source:
        results_dict = brute_force_single_source(wn, start_hour_of_pollution=18, end_hour_of_pollution=20)
        filename = "brute_force_single_results_18-20_window_" + inp_file[9:] + ".txt"
    else:
        results_dict = brute_force_two_source(wn, start_hour_of_pollution=18, end_hour_of_pollution=20)
        filename = "brute_force_double_results_18-20_window_" + inp_file[9:] + ".txt"

    # Save to textfile
    with open(filename, 'w') as results_file:
        for key, value in results_dict.items():
            results_file.write('%s:%s\n' % (key, value))
    print('Brute force 18-20 done!')


def pipe_diameter_algorithm(water_network):
    wn_dict = water_network.to_dict()


# TODO popraw wielowątkowość - nie odpalają się kolejne wątki
def brute_force_chemical_pollution_two_source_multithreaded(water_network):
    # Start time measure
    start_time = perf_counter()
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_THREADS)
    water_network_dict = wntr.network.to_dict(water_network)

    dataframe_structure = {
        'Node': [],
        'Simulation end polluted nodes [%]': [],
        'Max polluted nodes [%]': []
    }
    df_length = len(water_network_dict['nodes'])
    pollution_results = pd.DataFrame(index=range(df_length), columns=range(3))
    # Running simulations of pollution for all the nodes
    for i in range(len(water_network_dict['nodes'])):
        for j in range(len(water_network_dict['nodes'])):
            pool.submit(pollution_analysis_worker(water_network_dict, i, j, water_network, dataframe_structure,
                                                  pollution_results))

    pool.shutdown(wait=True)

    # Prepare results
    pollution_results_alltime = copy.deepcopy(pollution_results)
    pollution_results_alltime = pollution_results_alltime.sort_values(by=['Max polluted nodes [%]'], ascending=False)
    pollution_results = pollution_results.sort_values(by=['Simulation end polluted nodes [%]'], ascending=False)
    # Stop time measure
    end_time = perf_counter()
    duration = end_time - start_time

    return_dict = {
        'Alltime best node': pollution_results_alltime.iloc[0]['Node'],
        'Alltime best pollution [%]': pollution_results_alltime.iloc[0]['Max polluted nodes [%]'],
        'Sim end best node': pollution_results.iloc[0]['Node'],
        'Sim end best pollution [%]': pollution_results_alltime.iloc[0]['Simulation end polluted nodes [%]'],
        'Duration': duration
    }
    return return_dict


def pollution_analysis_worker(wn_dict, i, j, water_n, df_structure, poll_results):
    temp = len(wn_dict['nodes']) * i + j
    print("Started Thread: " + str(temp))

    t_water_n = copy.deepcopy(water_n)
    t_nodes_to_pollute = [wn_dict['nodes'][i]['name'], wn_dict['nodes'][j]['name']]
    analysis_results = pollution_analysis(t_water_n, t_nodes_to_pollute, i, j)
    df_structure['Node'] = t_nodes_to_pollute[0] + ' ' + t_nodes_to_pollute[1]
    # df_structure['Simulation end polluted nodes [%]'] = analysis_results['Simulation end polluted nodes [%]']
    # df_structure['Max polluted nodes [%]'] = analysis_results['Max polluted nodes [%]']
    index = len(poll_results) * i + j
    poll_results.loc[index] = df_structure
    del water_n


if __name__ == '__main__':
    get_results_bruteforce("networks/Net1.inp")
    print('Net1 brute force done!')
    get_results_bruteforce("networks/Net3.inp")
    print('Net3 brute force done!')
    get_results_bruteforce("networks/LongTermImprovement.inp")
    print('LTI brute force done!')

    get_results_bruteforce(water_network_path="networks/Net1.inp", two_source=True)
    print('Net1 brute force done!')
    get_results_bruteforce(water_network_path="networks/Net3.inp", two_source=True)
    print('Net3 brute force done!')

    # get_results_bruteforce(water_network_path="networks/LongTermImprovement.inp", two_source=True)
    # print('LTI brute force done!')


