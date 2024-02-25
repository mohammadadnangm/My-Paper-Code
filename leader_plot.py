import matplotlib.pyplot as plt
import pandas as pd


# Load the vehicle data
vehicle_data = pd.read_csv('selected_vehicle_data.csv')

# Create a scatter plot of the vehicles
plt.scatter(vehicle_data['longitude'], vehicle_data['latitude'], label='Vehicles')

# Highlight the leader
leader = vehicle_data[vehicle_data['group_leader'] == True]
plt.scatter(leader['longitude'], leader['latitude'], color='red', label='Leader')

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Vehicle Locations')
plt.legend()

# Display the plot
plt.show()