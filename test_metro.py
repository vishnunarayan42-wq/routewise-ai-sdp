
from metro_planner import get_metro_journeys

# Test a journey from Aluva to Edapally
journeys = get_metro_journeys("Aluva", "Edapally")

if journeys.empty:
    print("No scheduled journey found for this station pair.")
else:
    print("Metro journeys found:")
    print(journeys.head(10).to_string(index=False))
