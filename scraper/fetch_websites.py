import json
import os
import requests
from xml.etree import ElementTree
from collections import defaultdict
import itertools
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Brave/1.61.109.0" # Represents Brave Browser
DEFAULT_OPERATING_HOURS = "10:00 AM - 11:00 PM"
DEFAULT_CONTACT_INFO = "+91 9876543210"
MAX_LOCATIONS_PER_RESTAURANT_NAME = 5
MAX_UNIQUE_RESTAURANT_CHAINS = 10

def fetch_sitemap_links(target_sitemap_url: str) -> list[str]:
    """
    Fetches and parses an XML sitemap to extract all contained URLs.

    Args:
        target_sitemap_url: The URL of the XML sitemap.

    Returns:
        A list of URLs extracted from the sitemap, or an empty list on error.
    """
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    extracted_links = []
    try:
        http_response = requests.get(target_sitemap_url, headers=request_headers, timeout=30)
        http_response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        xml_root = ElementTree.fromstring(http_response.content)
        # Find all 'url' elements regardless of namespace, then find the 'loc' element within each
        # The '{*}' syntax handles namespaces gracefully.
        for url_element in xml_root.findall('{*}url'):
            loc_element = url_element.find('{*}loc')
            if loc_element is not None and loc_element.text:
                extracted_links.append(loc_element.text.strip())

        logging.info(f"Found {len(extracted_links)} URLs in sitemap: {target_sitemap_url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error fetching sitemap {target_sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        logging.error(f"XML parsing error for {target_sitemap_url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during sitemap parsing: {e}")

    return extracted_links

def parse_info_from_url(page_url: str) -> dict | None:
    """
    Attempts to extract basic restaurant information based on URL path segments.
    Assumes a structure like '.../restaurant-name/location-name'.

    Args:
        page_url: The URL of the restaurant page.

    Returns:
        A dictionary containing extracted info (name, url, location, notes),
        or None if parsing fails.
    """
    try:
        # Remove potential trailing slash and split
        url_segments = page_url.strip('/').split('/')
        # Check if we have enough segments (e.g., https://domain/path/name/location)
        if len(url_segments) >= 5:
            # Assume name is the 4th segment (index 3) and location is 5th (index 4)
            restaurant_name = url_segments[-2].replace('-', ' ').title()
            location_name = url_segments[-1].replace('-', ' ').title()
            return {
                "name": restaurant_name,
                "url": page_url,
                "location": location_name,
                "notes": "Basic info extracted from URL. Needs detailed scraping.",
                # Default time/contact will be added later if needed
            }
        else:
            logging.warning(f"URL structure insufficient for parsing: {page_url}")
            return None
    except IndexError:
        logging.warning(f"Index error while parsing URL segments for: {page_url}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing URL {page_url}: {e}")
        return None


def aggregate_locations_by_restaurant(link_list: list[str]) -> dict[str, list[dict]]:
    """
    Groups parsed restaurant location data by restaurant name, limiting locations per name.

    Args:
        link_list: A list of URLs to process.

    Returns:
        A dictionary where keys are restaurant names and values are lists of
        location data dictionaries (up to MAX_LOCATIONS_PER_RESTAURANT_NAME).
    """
    restaurants_by_name = defaultdict(list)
    parsed_location_count = 0
    for link in link_list:
        location_info = parse_info_from_url(link)
        if location_info:
            restaurant_name = location_info["name"]
            # Only add if we haven't reached the max limit for this restaurant
            if len(restaurants_by_name[restaurant_name]) < MAX_LOCATIONS_PER_RESTAURANT_NAME:
                restaurants_by_name[restaurant_name].append(location_info)
                parsed_location_count += 1

    logging.info(f"Aggregated {parsed_location_count} locations under {len(restaurants_by_name)} unique restaurant names.")
    return dict(restaurants_by_name) # Convert back to regular dict if defaultdict behavior is not needed downstream


def persist_restaurant_data(restaurant_entries: list[dict], output_filepath: str):
    """
    Saves the list of restaurant location data to a JSON file, ensuring default fields.

    Args:
        restaurant_entries: A list of dictionaries, each representing a restaurant location.
        output_filepath: The path to the JSON file to be created or overwritten.
    """
    output_structure = {"sites": []}

    for record in restaurant_entries:
        # Ensure required fields exist, adding defaults if necessary
        record.setdefault("Time", DEFAULT_OPERATING_HOURS)
        record.setdefault("contact", DEFAULT_CONTACT_INFO)
        output_structure["sites"].append(record)

    try:
        # Ensure the target directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir: # Avoid error if path is just a filename in the current dir
             os.makedirs(output_dir, exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Use ensure_ascii=False for broader character support (e.g., non-English names)
            json.dump(output_structure, outfile, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved {len(output_structure['sites'])} entries to {output_filepath}")
    except IOError as e:
        logging.error(f"Error writing to JSON file {output_filepath}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON persistence: {e}")


def process_sitemap_and_generate_config():
    """
    Main workflow: Fetches sitemap, parses URLs, aggregates data,
    selects a subset, and saves to a configuration file.
    """
    # Configuration
    source_sitemap_url = "https://www.eatsure.com/sitemaps/brands.xml"
    script_directory = os.path.dirname(__file__)
    output_json_path = os.path.join(script_directory, 'config', 'sites.json')

    # 1. Fetch URLs from the sitemap
    sitemap_links = fetch_sitemap_links(source_sitemap_url)
    if not sitemap_links:
        logging.warning("No links retrieved from sitemap. Halting process.")
        return

    # 2. Aggregate locations by restaurant name from parsed URLs
    aggregated_restaurants = aggregate_locations_by_restaurant(sitemap_links)
    if not aggregated_restaurants:
        logging.warning("No restaurant data could be aggregated. Halting process.")
        return

    # 3. Select a limited number of unique restaurant chains
    final_restaurant_selection = []
    # Use itertools.islice to take the first N items (restaurant names and their locations)
    for _, location_list in itertools.islice(aggregated_restaurants.items(), MAX_UNIQUE_RESTAURANT_CHAINS):
        # Extend the final list with all locations (up to MAX_LOCATIONS_PER_RESTAURANT_NAME) for the selected chains
        final_restaurant_selection.extend(location_list)

    if not final_restaurant_selection:
        logging.warning("No restaurants remained after selection. Halting process.")
        return

    # 4. Persist the selected data to the JSON configuration file
    persist_restaurant_data(final_restaurant_selection, output_json_path)

    logging.info(f"Sitemap processing complete. Output generated at {output_json_path}.")


if __name__ == "__main__":
    process_sitemap_and_generate_config()