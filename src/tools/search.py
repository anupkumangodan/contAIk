import json
import os
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        name, value = stripped.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name, value)


_load_dotenv()

# RESTAURANT SEARCH TOOL
class RestaurantSearchInput(BaseModel):
    """Input schema for restaurant search."""

    location: str = Field(..., description="location e.g San Francisco")
    cuisine: str = Field(..., description="Cuisine type e.g. Italian, Thai, Indian")
    rating: float = Field(..., description="Minimum rating e.g. 4.5")


@tool(args_schema=RestaurantSearchInput)
def search_restaurants(
    location: str,
    cuisine: str,
    rating: float,
) -> str:
    """Search for restaurants."""
    try:
        from serpapi import GoogleSearch
    except ImportError:
        return json.dumps({
            "error": "Missing dependency: install google-search-results for serpapi"
        })

    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY")
    if not api_key:
        return json.dumps({"error": "Missing SERPAPI_API_KEY or SERP_API_KEY"})

    params = {
        "api_key": api_key,
        "engine": "google_maps",
        "q": f"{cuisine} restaurants in {location}",
        "type": "search",
        "hl": "en",
    }

    try:
        results = GoogleSearch(params).get_dict()
        restaurants = results.get("local_results", [])

        if not restaurants:
            return json.dumps({"message": "No restaurants found"})

        output = []
        for restaurant in restaurants:
            restaurant_rating = restaurant.get("rating")
            if restaurant_rating is not None and float(restaurant_rating) < rating:
                continue

            output.append({
                "name": restaurant.get("title"),
                "rating": restaurant_rating,
                "reviews": restaurant.get("reviews"),
                "address": restaurant.get("address"),
                "price": restaurant.get("price"),
                "type": restaurant.get("type"),
                "phone": restaurant.get("phone"),
                "website": restaurant.get("website"),
                "link": restaurant.get("link"),
            })

            if len(output) == 10:
                break

        if not output:
            return json.dumps({
                "message": f"No {cuisine} restaurants found in {location} with rating >= {rating}"
            })

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

search_tools = [search_restaurants]
