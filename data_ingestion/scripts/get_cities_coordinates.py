import csv
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut

def get_city_coordinates(city_name, retries=3):
    geolocator = Nominatim(user_agent="city_locator_script", timeout=10)  # Increased timeout (seconds)

    for attempt in range(retries):
        try:
            location = geolocator.geocode(city_name + ", Morocco")
            if location:
                return location.latitude, location.longitude
            else:
                print(f"Could not find coordinates for {city_name}")
                return None, None
        except GeocoderTimedOut:
            print(f"Geocoding service timed out for {city_name}. Retrying... ({attempt+1}/{retries})")
            time.sleep(2)  # Wait before retrying
    print(f"Failed to get coordinates for {city_name} after {retries} attempts.")
    return None, None

def lookup_cities_and_store(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        cities_data = list(reader)

    # Open the output file to write the results
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['Collectivités territoriales', 'Population', 'region', 'Latitude', 'Longitude']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for city_data in cities_data:
            city_name = city_data['Collectivités territoriales']
            lat, lon = get_city_coordinates(city_name)
            if lat and lon:
                print(f"Found coordinates for {city_name}: {lat}, {lon}")
                city_data['Latitude'] = lat
                city_data['Longitude'] = lon
            else:
                city_data['Latitude'] = "Not Found"
                city_data['Longitude'] = "Not Found"

            writer.writerow(city_data)

            # To prevent hitting OSM's rate limits, add a delay
            time.sleep(1)

if __name__ == "__main__":
    input_file = "/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/population.csv"  # Input file containing city names
    output_file = "cities_with_coordinates.csv"  # Output file to save city coordinates
    lookup_cities_and_store(input_file, output_file)
    print(f"Coordinates have been saved to {output_file}")
