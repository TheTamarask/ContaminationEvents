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
NUMBER_OF_ITERATIONS = 5
POLLUTION_THRESHOLD = 300

# Create a water network model and dictionary
inp_file = 'networks/LongTermImprovement.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
wn_dict = wntr.network.to_dict(wn)


# Amend the dictionary
#
#
#
# Add changes from dictionary to Water Network
wn=wntr.network.from_dict(wn_dict)


# Add dource of chemical pollution to the first joint from the dictionary
wn.options.quality.parameter = 'CHEMICAL'
polluted_node=wn_dict['nodes'][0]

# Set node used for tracing - not used in quality simluation for pollution
#   wn.options.quality.trace_node = '111'

# Create pattern for a new source of pollution
source_pattern = wntr.network.elements.Pattern.binary_pattern('SourcePattern', start_time=START_HOUR_OF_POLLUTION*HOUR,
                                                              end_time=END_HOUR_OF_POLLUTION*HOUR, duration=wn.options.time.duration,
                                                              step_size=wn.options.time.pattern_timestep)
wn.add_pattern('SourcePattern', source_pattern)

# Graph the network witout any additions
wntr.graphics.plot_interactive_network(wn, title="Water network system diagram")


# Running few simulations of pollution
dataframe_structure = {
    'iteration': [],
    'Node': [],
    'Number_of_polluted_nodes': []
}
pollution_results = pd.DataFrame(dataframe_structure)
i = 0
while i < NUMBER_OF_ITERATIONS:
    i += 1

    # Select node to be polluted
    polluted_node_index = np.random.randint(0,len(wn_dict['nodes']))
    polluted_node = wn_dict['nodes'][polluted_node_index]
    print(polluted_node['name'])
    # Create a new source of pollution
    wn.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'SourcePattern')

    # Simulate hydraulics
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Plot results on the network
    Simulation_data = results.node['quality'].loc[5 * 3600, :]
    ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=Simulation_data, node_size=10, title='Pollution in the system')

    # Extract the number of junctions that are polluted above the threshold
    polluted_nodes = Simulation_data.ge(POLLUTION_THRESHOLD)
    number_of_polluted_nodes = sum(polluted_nodes)
    temp_dict = {"iteration": i, "Node": polluted_node['name'], "Number_of_polluted_nodes": number_of_polluted_nodes}
    pollution_results.loc[len(pollution_results)] = temp_dict

print(pollution_results)
# pollution_results.sort_values(by="Number_of_polluted_nodes", ascending=False)
pollution_results.sort_values(
    by="Number_of_polluted_nodes",
    ascending=False
)
print(pollution_results)

