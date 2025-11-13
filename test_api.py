import os, requests
API_KEY = os.environ["GMAPS_API"]

# Central Brussels – should have Street View coverage
lat, lng = 50.8467, 4.3525
r = requests.get(
    "https://maps.googleapis.com/maps/api/streetview/metadata",
    params={"location": f"{lat},{lng}", "key": API_KEY}
)
print(r.status_code, r.text)
