import pandas as pd
import traci
import pytz
import datetime
import random


# Function to get current date and time
def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Karachi"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME


# Command to start SUMO GUI
sumoCmd = ["sumo-gui", "-c", "osm.sumocfg", "--start"]

try:
    traci.start(sumoCmd)
    traci.simulationStep()  # Start the simulation
except Exception as e:
    print(f"Failed to start SUMO GUI: {e}")
    exit(1)

data_accumulator = []  # List to accumulate data

# Define the size of each cell (in meters)
cell_size = 500  # 50m x 50m

# Create a data structure to hold the cells and the vehicles in each cell
cells = {}

seen_vehicles = set()  # Set to keep track of seen vehicles

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    # During the simulation, assign each vehicle to a cell based on its position
    for i in range(0, len(vehicles)):
        try:
            vehicle_id = vehicles[i]
            if vehicle_id in seen_vehicles:
                continue  # Skip if vehicle data is already collected

            # Collect required vehicle properties
            timestep = getdatetime()
            speed = round(traci.vehicle.getSpeed(vehicle_id) * 3.6, 2)  # Speed in km/h
            x, y = traci.vehicle.getPosition(vehicle_id)
            longitude, latitude = traci.simulation.convertGeo(x, y)  # Longitude and latitude
            laneid = traci.vehicle.getLaneID(vehicle_id)
            running_position = round(traci.vehicle.getDistance(vehicle_id), 2)  # Running position from start of the lane

            # Calculate the cell id based on the position
            cell_id = (int(x // cell_size), int(y // cell_size))

            # Generate random values for the new attributes
            acting_frequency = random.randint(1, 20)  # Random integer between 1 and 20
            transmission_power = random.uniform(0.1, 5.0)  # Random float between 0.1 and 5.0
            traffic_rules_obeyed = random.choice([True, False])  # Random boolean value
            processing_power = random.uniform(1.0, 3.0)  # Random float between 1.0 and 3.0
            available_storage = random.randint(1, 100)  # Random integer between 1 and 100

            # Create a dictionary with vehicle data
            veh_data = {
                'datetime': timestep,
                'vehicle_id': vehicle_id,
                'speed': speed,
                'longitude': longitude,
                'latitude': latitude,
                'lane_id': laneid,
                'running_position': running_position,
                'cell_id': cell_id,
                'acting_frequency': acting_frequency,
                'transmission_power': transmission_power,
                'traffic_rules_obeyed': traffic_rules_obeyed,
                'processing_power': processing_power,
                'available_storage': available_storage
            }

            # Assign the vehicle to the cell
            if cell_id not in cells:
                cells[cell_id] = []
            cells[cell_id].append(veh_data)

            # Append vehicle data to the accumulator list
            data_accumulator.append(veh_data)

            # Add the vehicle to the seen vehicles set
            seen_vehicles.add(vehicle_id)

        except Exception as e:
            print(f"Failed to collect data for vehicle {vehicle_id}: {e}")

# Convert the accumulated data to a DataFrame
vehicle_data = pd.DataFrame(data_accumulator)

# Sort the DataFrame by the 'cell_id' column
vehicle_data = vehicle_data.sort_values(by='cell_id')

# Save vehicle data to CSV file (outside the simulation loop)
vehicle_data.to_csv('vehicle_data.csv', index=False)

# Print the total number of vehicles
print(f"Total number of vehicles extracted by simulation: {len(vehicle_data)}")

# Print the cell IDs and the number of vehicles in each cell
for cell_id, vehicles in cells.items():
    print(f"Cell ID: {cell_id}, Number of vehicles: {len(vehicles)}")

# When finished reading the data, stop the simulation
traci.close()  # Close the TraCI connection
