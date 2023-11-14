
from random import random
import wntr
import wntr.network.controls as controls

#Constants
HOUR = 3600
HOURS_OF_SIMULATION = 10
START_HOUR_OF_POLLUTION = 4
END_HOUR_OF_POLLUTION = 5

# Create a water network model and dictionary
inp_file = 'networks/LongTermImprovement.inp'
wn = wntr.network.WaterNetworkModel(inp_file)
wn_dict = wntr.network.to_dict(wn)


#Amend the dictionary
#
#
#
#Add changes from dictionary to Water Network
wn=wntr.network.from_dict(wn_dict)


#Add dource of chemical pollution to the first joint from the dictionary
wn.options.quality.parameter = 'CHEMICAL'
polluted_node=wn_dict['nodes'][0]


#Set node used for tracing - not used in quality simluation for pollution
#   wn.options.quality.trace_node = '111'


#Create pattern for a new source of pollution
source_pattern = wntr.network.elements.Pattern.binary_pattern('SourcePattern', start_time=START_HOUR_OF_POLLUTION*HOUR,
                                                              end_time=END_HOUR_OF_POLLUTION*HOUR, duration=wn.options.time.duration,
                                                              step_size=wn.options.time.pattern_timestep)
#Create a new source of pollution
wn.add_pattern('SourcePattern', source_pattern)
print(polluted_node['name'])
wn.add_source('PollutionSource', polluted_node['name'], 'SETPOINT', 2000, 'SourcePattern')

#Valve layer generation and plotting
#   random_valve_layer = wntr.network.generate_valve_layer(wn, 'random', 40, seed=123)
#   wntr.graphics.plot_valve_layer(wn, valve_layer=, add_colorbar=False)

# Graph the network witout any additions
wntr.graphics.plot_interactive_network(wn, title=wn.name)

# Simulate hydraulics
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Plot results on the network
Simulation_data = results.node['quality'].loc[5 * 3600, :]
ax1 = wntr.graphics.plot_interactive_network(wn, node_attribute=Simulation_data, node_size=10,
                           title='Pressure at 5 hours')