import re
import json
import os
from bs4 import BeautifulSoup
from utils import fetch_data, handle_errors # Assuming utils.py is in src directory

class RestaurantScraper:
    """Scrapes data for a single restaurant URL."""

    def __init__(self, url):
        self.url = url

    def scrape(self):
        """Fetches and parses restaurant data."""
        print(f"Scraping {self.url}")
        try:
            html = fetch_data(self.url)
            if not html:
                print(f"Failed to fetch HTML for {self.url}")
                return None

            soup = BeautifulSoup(html, 'html.parser')
            extracted_data = {}

            # --- Attempt to find JSON in script tags ---
            json_regex = re.compile(r'\{.*?\}', re.DOTALL) # Find potential JSON objects
            scripts = soup.find_all('script')
            menu_items = []
            for script in scripts:
                if script.string: # Check if script tag has content
                    potential_matches = json_regex.findall(script.string)
                    for match in potential_matches:
                        try:
                            parsed_json = json.loads(match)
                            if isinstance(parsed_json, dict) and 'product_name' in parsed_json:
                                menu_item = {
                                    "product_id": parsed_json.get("product_id"),
                                    "product_name": parsed_json.get("product_name"),
                                    "hsn_code": parsed_json.get("hsn_code"),
                                    "benefits": parsed_json.get("benefits"),
                                    "product_category_id": parsed_json.get("product_category_id"),
                                    "small_description": parsed_json.get("small_description"),
                                    "big_description": parsed_json.get("big_description"),
                                    "is_veg": parsed_json.get("is_veg"),
                                    "is_customizable": parsed_json.get("is_customizable"),
                                    "is_customizable_group": parsed_json.get("is_customizable_group"),
                                    "customization_limit": parsed_json.get("customization_limit"),
                                    "spice_level": parsed_json.get("spice_level"),
                                    "bought_count": parsed_json.get("bought_count"),
                                    "rating": parsed_json.get("rating"),
                                    "count_of_rating": parsed_json.get("count_of_rating"),
                                    "is_available": parsed_json.get("is_available"),
                                    "is_active": parsed_json.get("is_active"),
                                    "is_back_calculate_tax": parsed_json.get("is_back_calculate_tax"),
                                    "tax_category": parsed_json.get("tax_category"),
                                    "price": parsed_json.get("price"),
                                    "details": parsed_json.get("details"),
                                    "feature_tags": parsed_json.get("feature_tags"),
                                    "preparation_time": parsed_json.get("preparation_time"),
                                    "tags": parsed_json.get("tags"),
                                    "promo_tags": parsed_json.get("promo_tags"),
                                    "ml_tags": parsed_json.get("ml_tags"),
                                    "offer_tags": parsed_json.get("offer_tags"),
                                    "is_featured": parsed_json.get("is_featured"),
                                    "brand_name": parsed_json.get("brand_name"),
                                    "display_price": parsed_json.get("display_price"),
                                    "share": parsed_json.get("share"),
                                    "product_feedback": parsed_json.get("product_feedback"),
                                    "switch_off_msg": parsed_json.get("switch_off_msg"),
                                    "brand_display_name": parsed_json.get("brand_display_name"),
                                    "price_without_tax": parsed_json.get("price_without_tax"),
                                    "tax_amount": parsed_json.get("tax_amount"),
                                }
                                menu_items.append(menu_item)
                        except json.JSONDecodeError:
                            # Ignore strings that look like JSON but aren't valid
                            pass

            # --- Load additional data from sites.json ---
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sites.json')
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    sites_data = json.load(f)
                    # print("Loaded sites.json data:", sites_data)  # Debugging: Print loaded data
                    for site in sites_data.get('sites', []):
                        # print(f"Matching {self.url} with {site['url']}")  # Debugging: Print URL comparison
                        if site['url'].rstrip('/') == self.url.rstrip('/'):  # Normalize URLs
                            extracted_data['restaurant_name'] = site.get('name', 'Unknown Name')
                            extracted_data['location'] = site.get('location', 'Unknown Location')
                            extracted_data['available_time'] = site.get('Time', 'Unknown Time')
                            extracted_data['contact'] = site.get('contact', 'Unknown Contact')
                            break
            except FileNotFoundError:
                print(f"Warning: sites.json not found at {config_path}")
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON from {config_path}")
            extracted_data['menu_items'] = menu_items

            return extracted_data

        except Exception as e:
            handle_errors(f"Error during scraping {self.url}: {e}")
            return None