import matplotlib.pyplot as plt
import pandas as pd

# Load the updated DataFrame from the CSV file
vehicle_data = pd.read_csv('vehicle_data.csv')

# Get the count of vehicles in each cell
cell_vehicle_counts = vehicle_data['cell_id'].value_counts()

# Plot the graph
plt.figure(figsize=(10, 6))
plt.bar(cell_vehicle_counts.index, cell_vehicle_counts.values, color='blue')
plt.xlabel('Cell ID')
plt.ylabel('Number of Vehicles')
plt.title('Distribution of Vehicles in Cells')
plt.show()
