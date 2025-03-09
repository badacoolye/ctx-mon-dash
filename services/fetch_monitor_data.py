#!/usr/bin/env python
import os
import logging
import requests
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import argparse

# Import the direct token function from token_manager.py
from .token_manager import get_token_directly

# Third-party module to flexibly parse dates provided by the user.
from dateutil import parser as date_parser

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_all_records(session: requests.Session, initial_url: str) -> list:
    """
    Fetch all records from an API endpoint using pagination.
    
    :param session: The requests.Session() object with proper headers.
    :param initial_url: The URL to fetch data from.
    :return: A list of all records fetched.
    """
    all_records = []
    skip = 0
    page_size = 1000

    while True:
        # Update URL with the new $skip parameter value
        parsed = urlparse(initial_url)
        query_params = parse_qs(parsed.query)
        query_params['$skip'] = [str(skip)]
        new_query = urlencode(query_params, doseq=True)
        current_url = urlunparse(parsed._replace(query=new_query))
        
        logger.info(f"Fetching records with $skip={skip}")
        try:
            response = session.get(current_url)
            response.raise_for_status()
            data = response.json()

            # Optionally log the total count if available
            if skip == 0 and '@odata.count' in data:
                total_count = data['@odata.count']
                logger.info(f"Total records to fetch: {total_count}")

            records = data.get("value", [])
            if not records:
                break

            all_records.extend(records)
            logger.info(f"Fetched {len(records)} records; Total so far: {len(all_records)}")

            # Break if fewer than page_size records were returned
            if len(records) < page_size:
                break

            skip += page_size

        except requests.exceptions.RequestException as e:
            logger.error("Error fetching data", exc_info=True)
            raise

    return all_records

def fetch_and_save_data(session: requests.Session, url: str, output_prefix: str, output_dir: str = "output") -> pd.DataFrame:
    """
    Fetch data from the given URL and save it to a CSV file.
    
    :param session: The requests.Session() object with proper headers.
    :param url: The API endpoint URL.
    :param output_prefix: A prefix for the output CSV file.
    :param output_dir: The base directory where output files will be stored.
    :return: A pandas DataFrame containing the fetched data.
    """
    try:
        records = fetch_all_records(session, url)
        logger.info(f"Fetched a total of {len(records)} records for '{output_prefix}'.")
        df = pd.json_normalize(records, sep='_')

        # Create output directory with today's date
        current_subfolder = datetime.now().strftime("%Y%m%d")
        full_output_dir = os.path.join(output_dir, current_subfolder)
        os.makedirs(full_output_dir, exist_ok=True)

        # Create a unique filename using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(full_output_dir, f"{output_prefix}_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to: {output_file}")
        return df

    except Exception as e:
        logger.error("Error fetching and saving data", exc_info=True)
        raise

def main():
    # Parse command-line arguments for start and end dates.
    parser = argparse.ArgumentParser(description="Fetch data with date filtering.")
    parser.add_argument("--start-date", required=True,
                        help="Start date for filtering records (e.g. '2024-12-31 23:00:01', 'Dec 31, 2024 11:00 PM', etc.)")
    parser.add_argument("--end-date", required=True,
                        help="End date for filtering records (e.g. '2025-01-31 22:59:59', 'Jan 31, 2025 10:59 PM', etc.)")
    args = parser.parse_args()

    # Parse the user-provided dates.
    try:
        start_date = date_parser.parse(args.start_date)
        end_date = date_parser.parse(args.end_date)
    except Exception as e:
        logger.error("Error parsing date parameters. Please provide valid dates.", exc_info=True)
        return

    # If the dates are naive (no timezone), assume UTC.
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Format dates as required: "YYYY-MM-DDTHH:MM:SS.000Z"
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    logger.info(f"Using start-date: {start_date_str} and end-date: {end_date_str}")

    # Read client credentials from environment variables
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.error("Client credentials not set in .env file. Please set CLIENT_ID and CLIENT_SECRET.")
        return

    # Get the token directly from the API
    try:
        token = get_token_directly(client_id, client_secret)
        logger.info(f"Received token: {token[:20]}...")  # Display a preview of the token
    except Exception as e:
        logger.error("Failed to obtain token", exc_info=True)
        return

    # Get Citrix Customer ID from environment variables
    customer_id = os.getenv("CITRIX_CUSTOMER_ID")
    if not customer_id:
        logger.error("CITRIX_CUSTOMER_ID not set in .env file.")
        return

    # Create a requests session and set the necessary headers
    session = requests.Session()
    session.headers.update({
        "Authorization": f"CwsAuth bearer={token}",
        "Accept": "application/json",
        "Citrix-CustomerId": customer_id
    })

    # Define common query parameters for the API call.
    base_params = {
        "$select": "TotalLaunchesCount,SummaryDate,TotalUsageDuration,PeakConcurrentInstanceCount",
        "$filter": f"(SummaryDate ge cast({start_date_str}, Edm.DateTimeOffset)) and (SummaryDate le cast({end_date_str}, Edm.DateTimeOffset))",
        "$orderby": "TotalUsageDuration asc",
        "$count": "true",
        "$skip": "0"
    }
    base_api_url = "https://api.cloud.com/monitorodata"
    urls = {
        "applications": f"{base_api_url}/ApplicationActivitySummaries?{urlencode(base_params)}&$expand=Application($select=Name)",
        "servers": f"{base_api_url}/ServerOSDesktopSummaries?{urlencode(base_params)}&$expand=DesktopGroup($select=Name)"
    }

    # Fetch data for each endpoint and save it as CSV
    try:
        for data_type, url in urls.items():
            logger.info(f"Fetching {data_type} data...")
            fetch_and_save_data(session, url, data_type)
        logger.info("Data fetching completed successfully.")
    except Exception as e:
        logger.error("An error occurred during data fetching", exc_info=True)

if __name__ == "__main__":
    main()