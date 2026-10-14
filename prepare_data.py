import pandas as pd

# Load the GTFS routes file
routes = pd.read_csv("data/routes.txt")

# Keep only buses
# GTFS route_type 3 = Bus
bus_routes = routes[routes["route_type"] == 3].copy()

# Save bus-only routes
bus_routes.to_csv(
    "data/bus_routes.csv",
    index=False
)

# Show results
print("Total routes:", len(routes))
print("Bus routes:", len(bus_routes))

print("\nFirst 10 bus routes:")
print(
    bus_routes[
        [
            "route_id",
            "agency_id",
            "route_short_name",
            "route_long_name",
            "route_type"
        ]
    ].head(10)
)

print("\nBus data saved successfully!")