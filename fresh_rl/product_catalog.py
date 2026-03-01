"""
Product catalog: 7 categories × 15 SKUs = 105 generated + 5 legacy = 110 products.

Provides reproducible SKU generation with realistic economics for fresh food
markdown channel simulation.

Usage:
    from fresh_rl.product_catalog import generate_catalog, get_product_names, get_profile

    catalog = generate_catalog()          # 110 products, cached
    names = get_product_names("meats")    # 15 meat SKU names
    profile = get_profile("salmon_fillet") # dict ready for MarkdownProductEnv
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CategorySpec:
    """Specification for a product category with parameter ranges."""
    name: str
    markdown_window_options: List[int]  # hours, cycled by SKU index
    price_range: tuple  # (min, max) base_price
    elasticity_range: tuple  # (min, max) price_elasticity
    cost_fraction_range: tuple  # (min, max) as fraction of base_price
    demand_range: tuple  # (min, max) base_markdown_demand
    inventory_range: tuple  # (min, max) initial_inventory
    sku_names: List[str] = field(default_factory=list)


# ── Category definitions ──────────────────────────────────────────────────

CATEGORIES: Dict[str, CategorySpec] = {
    "meats": CategorySpec(
        name="meats",
        markdown_window_options=[24],
        price_range=(5.0, 15.0),
        elasticity_range=(2.5, 3.5),
        cost_fraction_range=(0.45, 0.55),
        demand_range=(2.5, 5.0),
        inventory_range=(10, 25),
        sku_names=[
            "ground_beef_1lb", "chicken_breast", "pork_chop", "beef_steak_ribeye",
            "lamb_chop", "bacon_pack", "turkey_breast", "beef_roast_chuck",
            "pork_tenderloin", "chicken_thigh_pack", "italian_sausage",
            "ground_turkey_1lb", "beef_stew_meat", "pork_ribs_rack", "chicken_wings_2lb",
        ],
    ),
    "seafood": CategorySpec(
        name="seafood",
        markdown_window_options=[12, 24],
        price_range=(7.0, 20.0),
        elasticity_range=(2.0, 3.0),
        cost_fraction_range=(0.45, 0.60),
        demand_range=(2.0, 4.0),
        inventory_range=(8, 20),
        sku_names=[
            "salmon_fillet", "shrimp_1lb", "sushi_combo_6pc", "poke_bowl",
            "lobster_tail", "tilapia_fillet", "crab_cakes_2pk", "tuna_steak",
            "cod_fillet", "mussels_2lb", "smoked_salmon_4oz", "scallops_8oz",
            "clam_chowder_qt", "shrimp_cocktail", "fish_tacos_kit",
        ],
    ),
    "vegetables": CategorySpec(
        name="vegetables",
        markdown_window_options=[24, 48],
        price_range=(1.50, 5.0),
        elasticity_range=(3.0, 4.5),
        cost_fraction_range=(0.30, 0.45),
        demand_range=(4.0, 8.0),
        inventory_range=(15, 35),
        sku_names=[
            "salad_mix_5oz", "spinach_bunch", "mushroom_8oz", "asparagus_bunch",
            "bell_pepper_3pk", "broccoli_crown", "cherry_tomatoes_pt", "green_beans_lb",
            "zucchini_2pk", "corn_on_cob_4pk", "kale_bunch", "brussels_sprouts_lb",
            "cauliflower_head", "snap_peas_8oz", "herb_mix_fresh",
        ],
    ),
    "fruits": CategorySpec(
        name="fruits",
        markdown_window_options=[24, 48],
        price_range=(2.0, 7.0),
        elasticity_range=(3.0, 4.0),
        cost_fraction_range=(0.30, 0.45),
        demand_range=(4.0, 7.0),
        inventory_range=(15, 30),
        sku_names=[
            "strawberries_1lb", "blueberries_6oz", "cut_watermelon", "avocado_3pk",
            "raspberries_6oz", "mango_sliced", "grapes_red_2lb", "pineapple_chunks",
            "peaches_4pk", "kiwi_6pk", "banana_bunch_organic", "blackberries_6oz",
            "cantaloupe_half", "mixed_berries_12oz", "pomegranate_seeds_8oz",
        ],
    ),
    "dairy": CategorySpec(
        name="dairy",
        markdown_window_options=[48],
        price_range=(1.50, 6.0),
        elasticity_range=(2.5, 3.5),
        cost_fraction_range=(0.25, 0.40),
        demand_range=(4.0, 8.0),
        inventory_range=(15, 35),
        sku_names=[
            "yogurt_greek_plain", "milk_whole_gallon", "fresh_mozzarella",
            "cottage_cheese_16oz", "cream_cheese_8oz", "butter_unsalted",
            "sour_cream_16oz", "heavy_cream_pt", "yogurt_strawberry",
            "cheddar_block_8oz", "ricotta_15oz", "half_and_half_qt",
            "whipped_cream_8oz", "goat_cheese_4oz", "brie_wheel_8oz",
        ],
    ),
    "bakery": CategorySpec(
        name="bakery",
        markdown_window_options=[24],
        price_range=(2.50, 7.0),
        elasticity_range=(3.5, 4.5),
        cost_fraction_range=(0.30, 0.45),
        demand_range=(3.5, 6.0),
        inventory_range=(12, 25),
        sku_names=[
            "sourdough_loaf", "croissants_4pk", "bagels_6pk", "cinnamon_rolls_4pk",
            "baguette_french", "dinner_rolls_12pk", "multigrain_loaf",
            "muffins_blueberry_4pk", "danish_pastry_4pk", "rye_bread_loaf",
            "brioche_buns_6pk", "scones_cranberry_4pk", "focaccia_herb",
            "challah_loaf", "tortillas_flour_10pk",
        ],
    ),
    "deli_prepared": CategorySpec(
        name="deli_prepared",
        markdown_window_options=[12, 24],
        price_range=(5.0, 14.0),
        elasticity_range=(2.5, 3.5),
        cost_fraction_range=(0.35, 0.50),
        demand_range=(2.5, 5.0),
        inventory_range=(8, 20),
        sku_names=[
            "rotisserie_chicken", "deli_sandwich_club", "soup_chicken_noodle_qt",
            "mac_and_cheese_lb", "caesar_salad_kit", "sushi_roll_california",
            "pizza_slice_pepperoni", "fried_chicken_8pc", "pasta_salad_lb",
            "hummus_roasted_garlic", "quiche_lorraine", "spring_rolls_6pk",
            "pulled_pork_lb", "meatball_sub", "greek_salad_bowl",
        ],
    ),
}

# ── Legacy profiles (verbatim from environment.py) ────────────────────────

LEGACY_PROFILES: Dict[str, dict] = {
    "salad_mix": {
        "markdown_window_hours": 24,
        "initial_inventory": 25,
        "base_price": 3.50,
        "base_markdown_demand": 5.0,
        "price_elasticity": 3.5,
        "cost_per_unit": 1.20,
    },
    "fresh_chicken": {
        "markdown_window_hours": 24,
        "initial_inventory": 15,
        "base_price": 8.00,
        "base_markdown_demand": 3.0,
        "price_elasticity": 2.8,
        "cost_per_unit": 4.00,
    },
    "yogurt": {
        "markdown_window_hours": 48,
        "initial_inventory": 30,
        "base_price": 2.50,
        "base_markdown_demand": 6.0,
        "price_elasticity": 3.0,
        "cost_per_unit": 0.80,
    },
    "bakery_bread": {
        "markdown_window_hours": 24,
        "initial_inventory": 18,
        "base_price": 4.00,
        "base_markdown_demand": 4.5,
        "price_elasticity": 4.0,
        "cost_per_unit": 1.50,
    },
    "sushi": {
        "markdown_window_hours": 12,
        "initial_inventory": 10,
        "base_price": 10.00,
        "base_markdown_demand": 2.5,
        "price_elasticity": 2.5,
        "cost_per_unit": 5.00,
    },
}

# ── SKU generation ────────────────────────────────────────────────────────

# Keys that go to MarkdownProductEnv (the 6 env fields)
_ENV_KEYS = {
    "markdown_window_hours", "initial_inventory", "base_price",
    "base_markdown_demand", "price_elasticity", "cost_per_unit",
}


def generate_sku_profile(category: CategorySpec, sku_name: str, sku_index: int, rng: np.random.Generator) -> dict:
    """Generate a single SKU profile from category spec + seeded RNG."""
    base_price = rng.uniform(*category.price_range)
    cost_fraction = rng.uniform(*category.cost_fraction_range)
    elasticity = rng.uniform(*category.elasticity_range)
    demand = rng.uniform(*category.demand_range)
    inventory = int(rng.uniform(*category.inventory_range))
    window = category.markdown_window_options[sku_index % len(category.markdown_window_options)]

    return {
        "markdown_window_hours": window,
        "initial_inventory": inventory,
        "base_price": round(base_price, 2),
        "base_markdown_demand": round(demand, 1),
        "price_elasticity": round(elasticity, 2),
        "cost_per_unit": round(base_price * cost_fraction, 2),
        # Metadata (stripped by get_profile)
        "_category": category.name,
        "_sku_name": sku_name,
    }


# ── Catalog registry ─────────────────────────────────────────────────────

_catalog_cache: Optional[Dict[str, dict]] = None


def generate_catalog(seed: int = 42) -> Dict[str, dict]:
    """Generate the full product catalog (cached). Returns {name: profile_dict}."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    catalog: Dict[str, dict] = {}

    # Legacy products first
    for name, profile in LEGACY_PROFILES.items():
        catalog[name] = {**profile, "_category": "legacy", "_sku_name": name}

    # Generated SKUs
    rng = np.random.default_rng(seed)
    for cat_name, spec in CATEGORIES.items():
        for idx, sku_name in enumerate(spec.sku_names):
            if sku_name in catalog:
                continue  # skip if name collides with legacy
            catalog[sku_name] = generate_sku_profile(spec, sku_name, idx, rng)

    _catalog_cache = catalog
    return catalog


def get_product_names(category: Optional[str] = None) -> List[str]:
    """Get product names, optionally filtered by category (or 'legacy')."""
    catalog = generate_catalog()
    if category is None:
        return sorted(catalog.keys())
    return sorted(
        name for name, prof in catalog.items()
        if prof.get("_category") == category
    )


def get_profile(product_name: str) -> dict:
    """Get a single product's env-ready profile (no metadata keys)."""
    catalog = generate_catalog()
    if product_name not in catalog:
        available = ", ".join(sorted(catalog.keys())[:10])
        raise ValueError(
            f"Unknown product '{product_name}'. "
            f"Available: {available}, ... ({len(catalog)} total)"
        )
    return {k: v for k, v in catalog[product_name].items() if k in _ENV_KEYS}


def get_categories() -> List[str]:
    """Return list of category names (including 'legacy')."""
    return ["legacy"] + sorted(CATEGORIES.keys())


def print_catalog_summary():
    """Print a formatted catalog summary grouped by category."""
    catalog = generate_catalog()
    categories = get_categories()

    print(f"\n{'='*70}")
    print(f"  Product Catalog — {len(catalog)} products across {len(categories)} categories")
    print(f"{'='*70}")

    for cat in categories:
        names = get_product_names(cat)
        if not names:
            continue
        print(f"\n  [{cat.upper()}] ({len(names)} SKUs)")
        for name in names:
            p = catalog[name]
            print(
                f"    {name:<30s} "
                f"${p['base_price']:>6.2f}  "
                f"{p['markdown_window_hours']:>2d}h  "
                f"inv={p['initial_inventory']:>2d}  "
                f"e={p['price_elasticity']:.2f}  "
                f"cost=${p['cost_per_unit']:.2f}"
            )

    print(f"\n{'='*70}\n")
