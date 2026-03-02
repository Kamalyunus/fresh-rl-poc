"""
Product catalog: 7 categories × 21-22 SKUs = 150 products.

Provides reproducible SKU generation with realistic economics for fresh food
markdown channel simulation. All products use 24h markdown windows.

Usage:
    from fresh_rl.product_catalog import generate_catalog, get_product_names, get_profile

    catalog = generate_catalog()          # 150 products, cached
    names = get_product_names("meats")    # 22 meat SKU names
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
    pack_size_range: tuple = (1, 1)  # (min, max) units per package
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
        pack_size_range=(1, 4),
        sku_names=[
            "ground_beef_1lb", "chicken_breast", "pork_chop", "beef_steak_ribeye",
            "lamb_chop", "bacon_pack", "turkey_breast", "beef_roast_chuck",
            "pork_tenderloin", "chicken_thigh_pack", "italian_sausage",
            "ground_turkey_1lb", "beef_stew_meat", "pork_ribs_rack", "chicken_wings_2lb",
            "flank_steak", "chicken_drumsticks_4pk", "beef_short_ribs",
            "pork_shoulder_roast", "veal_cutlet", "bison_burger_2pk", "duck_breast",
        ],
    ),
    "seafood": CategorySpec(
        name="seafood",
        markdown_window_options=[24],
        price_range=(7.0, 20.0),
        elasticity_range=(2.0, 3.0),
        cost_fraction_range=(0.45, 0.60),
        demand_range=(2.0, 4.0),
        inventory_range=(8, 20),
        pack_size_range=(1, 2),
        sku_names=[
            "salmon_fillet", "shrimp_1lb", "sushi_combo_6pc", "poke_bowl",
            "lobster_tail", "tilapia_fillet", "crab_cakes_2pk", "tuna_steak",
            "cod_fillet", "mussels_2lb", "smoked_salmon_4oz", "scallops_8oz",
            "clam_chowder_qt", "shrimp_cocktail", "fish_tacos_kit",
            "swordfish_steak", "catfish_fillet", "oysters_dozen",
            "crawfish_2lb", "halibut_fillet", "anchovies_tin", "sardines_can",
        ],
    ),
    "vegetables": CategorySpec(
        name="vegetables",
        markdown_window_options=[24],
        price_range=(1.50, 5.0),
        elasticity_range=(3.0, 4.5),
        cost_fraction_range=(0.30, 0.45),
        demand_range=(4.0, 8.0),
        inventory_range=(15, 35),
        pack_size_range=(1, 4),
        sku_names=[
            "salad_mix_5oz", "spinach_bunch", "mushroom_8oz", "asparagus_bunch",
            "bell_pepper_3pk", "broccoli_crown", "cherry_tomatoes_pt", "green_beans_lb",
            "zucchini_2pk", "corn_on_cob_4pk", "kale_bunch", "brussels_sprouts_lb",
            "cauliflower_head", "snap_peas_8oz", "herb_mix_fresh",
            "artichoke_2pk", "radishes_bunch", "celery_stalks",
            "sweet_potato_3lb", "eggplant_each", "leek_bunch",
        ],
    ),
    "fruits": CategorySpec(
        name="fruits",
        markdown_window_options=[24],
        price_range=(2.0, 7.0),
        elasticity_range=(3.0, 4.0),
        cost_fraction_range=(0.30, 0.45),
        demand_range=(4.0, 7.0),
        inventory_range=(15, 30),
        pack_size_range=(1, 6),
        sku_names=[
            "strawberries_1lb", "blueberries_6oz", "cut_watermelon", "avocado_3pk",
            "raspberries_6oz", "mango_sliced", "grapes_red_2lb", "pineapple_chunks",
            "peaches_4pk", "kiwi_6pk", "banana_bunch_organic", "blackberries_6oz",
            "cantaloupe_half", "mixed_berries_12oz", "pomegranate_seeds_8oz",
            "cherries_1lb", "plums_4pk", "nectarines_4pk",
            "papaya_half", "lychee_8oz", "figs_6pk",
        ],
    ),
    "dairy": CategorySpec(
        name="dairy",
        markdown_window_options=[24],
        price_range=(1.50, 6.0),
        elasticity_range=(2.5, 3.5),
        cost_fraction_range=(0.25, 0.40),
        demand_range=(4.0, 8.0),
        inventory_range=(15, 35),
        pack_size_range=(1, 2),
        sku_names=[
            "yogurt_greek_plain", "milk_whole_gallon", "fresh_mozzarella",
            "cottage_cheese_16oz", "cream_cheese_8oz", "butter_unsalted",
            "sour_cream_16oz", "heavy_cream_pt", "yogurt_strawberry",
            "cheddar_block_8oz", "ricotta_15oz", "half_and_half_qt",
            "whipped_cream_8oz", "goat_cheese_4oz", "brie_wheel_8oz",
            "yogurt_vanilla", "milk_oat_half_gal", "feta_crumbles_6oz",
            "mascarpone_8oz", "kefir_strawberry_qt", "gruyere_wedge_6oz",
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
        pack_size_range=(1, 12),
        sku_names=[
            "sourdough_loaf", "croissants_4pk", "bagels_6pk", "cinnamon_rolls_4pk",
            "baguette_french", "dinner_rolls_12pk", "multigrain_loaf",
            "muffins_blueberry_4pk", "danish_pastry_4pk", "rye_bread_loaf",
            "brioche_buns_6pk", "scones_cranberry_4pk", "focaccia_herb",
            "challah_loaf", "tortillas_flour_10pk",
            "pretzel_rolls_6pk", "cornbread_loaf", "pita_bread_6pk",
            "banana_bread_loaf", "olive_bread_loaf", "pumpernickel_loaf",
        ],
    ),
    "deli_prepared": CategorySpec(
        name="deli_prepared",
        markdown_window_options=[24],
        price_range=(5.0, 14.0),
        elasticity_range=(2.5, 3.5),
        cost_fraction_range=(0.35, 0.50),
        demand_range=(2.5, 5.0),
        inventory_range=(8, 20),
        pack_size_range=(1, 8),
        sku_names=[
            "rotisserie_chicken", "deli_sandwich_club", "soup_chicken_noodle_qt",
            "mac_and_cheese_lb", "caesar_salad_kit", "sushi_roll_california",
            "pizza_slice_pepperoni", "fried_chicken_8pc", "pasta_salad_lb",
            "hummus_roasted_garlic", "quiche_lorraine", "spring_rolls_6pk",
            "pulled_pork_lb", "meatball_sub", "greek_salad_bowl",
            "chicken_tikka_bowl", "beef_empanadas_4pk", "cobb_salad_bowl",
            "teriyaki_chicken_bowl", "stuffed_peppers_2pk", "caprese_salad_bowl",
            "bbq_pulled_chicken_lb",
        ],
    ),
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

    # Generated SKUs
    rng = np.random.default_rng(seed)
    pack_rng = np.random.default_rng(seed + 10000)  # separate RNG for pack_size
    for cat_name, spec in CATEGORIES.items():
        for idx, sku_name in enumerate(spec.sku_names):
            profile = generate_sku_profile(spec, sku_name, idx, rng)
            # Add pack_size as metadata (does NOT affect env dynamics)
            profile["_pack_size"] = int(pack_rng.uniform(*spec.pack_size_range))
            catalog[sku_name] = profile

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
    """Return list of category names."""
    return sorted(CATEGORIES.keys())


def get_product_features(product_name: str, inventory_mult: float = 1.0) -> np.ndarray:
    """Return 4-dim [0,1] observable product features for state augmentation.

    Features (all normalized within category ranges):
        [0] price_norm       — base_price position in category range
        [1] cost_frac_norm   — cost fraction position in category range
        [2] inventory_norm   — initial_inventory (× mult) position in category range
        [3] pack_size_norm   — pack_size position in category range

    These are observable product attributes (printed on packaging, visible in
    store systems) — no simulator leakage (elasticity, base_demand excluded).
    """
    catalog = generate_catalog()
    if product_name not in catalog:
        raise ValueError(f"Unknown product '{product_name}'")

    profile = catalog[product_name]
    cat = CATEGORIES[profile["_category"]]

    def _norm(val, lo, hi):
        return float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    price_norm = _norm(profile["base_price"], *cat.price_range)
    cost_frac = profile["cost_per_unit"] / max(profile["base_price"], 1e-6)
    cost_frac_norm = _norm(cost_frac, *cat.cost_fraction_range)
    inv = profile["initial_inventory"] * inventory_mult
    inventory_norm = _norm(inv, cat.inventory_range[0], cat.inventory_range[1] * max(inventory_mult, 1.0))
    pack_size_norm = _norm(profile.get("_pack_size", 1), *cat.pack_size_range)

    return np.array([price_norm, cost_frac_norm, inventory_norm, pack_size_norm], dtype=np.float32)


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
