from datetime import datetime

# Simple rule-based recommendations with Sri Lanka context.
# Extendable to plug an LLM later.
class RecommendationEngine:
    BASE_RULES = {
        "plastic": [
            "Rinse and dry plastic items to avoid contamination.",
            "Check local PET/HDPE collection days; keep caps on if required.",
            "Avoid burning plastic; it releases toxic fumes."
        ],
        "paper": [
            "Keep paper clean and dry; remove any plastic films.",
            "Flatten boxes to save space.",
        ],
        "glass": [
            "Rinse glass; separate by color if your municipality requires.",
            "Do not put broken glass in open bags—wrap securely to protect workers."
        ],
        "metal": [
            "Rinse cans; crush if allowed.",
            "Collect aluminum separately for better resale value."
        ],
        "cardboard": [
            "Flatten and tie bundles with twine.",
            "Keep away from moisture to maintain recyclability."
        ],
        "trash": [
            "Securely bag residual waste.",
            "Avoid mixing with recyclables to reduce contamination."
        ]
    }

    REGION_NOTES = {
        "LK-11": "Western Province: Many LGAs run door-to-door dry waste collection 1–2 days/week.",
        "LK-1": "Sri Lanka: Follow your Pradeshiya Sabha/MC schedule for recyclables.",
    }

    def recommend(self, label: str, region: str = "LK-11", city: str = "Colombo") -> dict:
        rules = self.BASE_RULES.get(label, self.BASE_RULES["trash"])
        region_note = self.REGION_NOTES.get(region, self.REGION_NOTES["LK-1"])
        tip = f"{region_note} Check {city} Municipal Council notices for latest schedules."
        return {
            "label": label,
            "tips": rules,
            "local_guidance": tip,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
