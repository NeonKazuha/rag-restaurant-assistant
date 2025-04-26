import json
import os
import sys
import logging

# Adjust path to import from sibling directory 'scraper' and 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Add project root to sys.path if not already there

# Imports assuming 'scraper' and 'src' are modules relative to project_root
try:
    from scraper.web_scraper import RestaurantScraper # Corrected import path
    # Assuming load_config is in src/config_loader.py or similar, adjust if needed.
    # If it's truly in src/main.py, that might indicate main.py has multiple responsibilities.
    # For now, we keep the original assumption based on the input code.
except ImportError as e:
    logging.exception("Failed to import necessary modules. Check project structure and sys.path.")
    sys.exit(f"Import Error: {e}. Please ensure the script is run correctly relative to the project root.")


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_and_save_raw_data():
    """Extract raw data from restaurant URLs and save to raw_extracted_data.json."""
    sites_config = load_config() # Assumes load_config finds 'config/sites.json' relative to project root or CWD

    if not sites_config or 'sites' not in sites_config or not sites_config['sites']:
        logging.error('Error: No sites found in config/sites.json or the file is invalid.')

    all_extracted_data = []  # List to store all extracted data
    processed_count = 0
    error_count = 0

    logging.info(f"Starting extraction for {len(sites_config['sites'])} sites listed in config.")

    for site in sites_config['sites']:
        url = site.get('url')
        name = site.get('name', url)  # Use name if available, otherwise URL

        if not url:
            logging.warning(f"Skipping site entry without a URL: {site}")
            continue

        logging.info(f"Attempting to scrape: {name} ({url})")
        try:
            # Instantiate scraper from the corrected import
            scraper = RestaurantScraper(url)
            restaurant_data = scraper.scrape() # scrape() method handles its own internal errors/logging

            if restaurant_data:
                logging.info(f"Successfully extracted data for {name}")
                # Optional: Add site metadata from config if not already present from scraper
                restaurant_data['config_name'] = name
                restaurant_data['config_url'] = url
                all_extracted_data.append(restaurant_data)
                processed_count += 1
            else:
                # scrape() returns None on failure, which is already logged within scrape() or fetch_data()
                logging.warning(f"No data extracted for {name} ({url}). Check previous logs for details.")
                error_count += 1
        except Exception as error:
            # Catch exceptions during scraper instantiation or unexpected issues in scrape() call
            logging.error(f"Unhandled error scraping {name} ({url}): {error}", exc_info=True)
            # Optionally use the imported error handler
            # handle_errors(error, context=f"scraping orchestration for {url}")
            error_count += 1

    logging.info(f"Extraction finished. Successfully processed: {processed_count}, Failed/No Data: {error_count}")

    # Save all extracted data to raw_extracted_data.json (replacing existing data)
    # Ensure output directory exists relative to this script's location
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'raw_extracted_data.json')

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved extracted data for {processed_count} sites to {output_path}")
    except IOError as e:
        logging.error(f"Failed to write output file {output_path}: {e}")
    except TypeError as e:
         logging.error(f"Failed to serialize data to JSON: {e}")


if __name__ == "__main__":
    extract_and_save_raw_data()