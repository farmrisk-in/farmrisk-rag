"""
presowing_prompts.py — Central prompt definitions for all 7 Pre-Sowing Advisory sections.

HOW TO EDIT:
  All LLM instructions live in PRESOWING_SYSTEM_PROMPT and PRESOWING_USER_TEMPLATE below.
  - PRESOWING_SYSTEM_PROMPT: Sets the AI role, output format rules, and length constraints.
  - PRESOWING_USER_TEMPLATE: Per-request prompt. Uses .format(**kwargs) to inject inputs.

OUTPUT FORMAT:
  The LLM returns a valid JSON object with exactly 7 keys.
  Each value is a Markdown string rendered directly by the frontend parser.

MARKDOWN CONVENTIONS (consistent across all sections):
  **bold**          → key terms, product names, quantities
  *italic*          → timing, stage names, caveats
  | col | col |     → GFM tables
  - bullet          → lists
  > ⚠️ text        → warning callout (render red in UI)
  > 💡 text        → tip callout (render green in UI)
  ### heading       → sub-section heading within a card
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Sets the AI role and strict output contract.
# Edit this to change tone, format rules, or length constraints.
# ---------------------------------------------------------------------------

PRESOWING_SYSTEM_PROMPT = """You are an expert agricultural advisory system for Indian farmers.
You generate highly structured, crop-specific, state-specific pre-sowing advisories grounded
in data from ICAR, State Agricultural Universities (SAU), and Krishi Vigyan Kendras (KVK).

OUTPUT CONTRACT — follow these rules exactly:
1. Return ONLY a valid JSON object. No preamble, no markdown code fences, no extra text.
2. The JSON must have exactly these 7 keys:
   "sowing_window", "seed_selection", "field_preparation",
   "fertilizer_plan", "irrigation", "weed_management", "pest_disease"
3. Each value is a Markdown string. Use GitHub Flavored Markdown:
   - **bold** for key terms, rates, product names
   - *italic* for timing, growth stages, caveats
   - | Table | with | pipes | for structured data (always include header separator row)
   - > ⚠️ **Warning:** for critical hard rules (renders as red callout in UI)
   - > 💡 **Tip:** for useful tips (renders as green callout in UI)
   - Bullet lists with - for enumerations
4. STRICT LENGTH LIMITS per section (count carefully):
   - sowing_window:     80–120 words  (1 table with 3 cols, 1 warning blockquote)
   - seed_selection:    150–200 words (1 intro line, 1 filter table, bullet list, 1 blockquote)
   - field_preparation: 150–200 words (1 operation table: Operation | Timing | Detail)
   - fertilizer_plan:  180–220 words (baseline headline, 1 split table, bullet add-ons, 1 warning)
   - irrigation:       130–170 words (1 summary line, 1 stage table, 1 warning blockquote)
   - weed_management:  80–120 words  (1 table: Stage | Action | Product & dose, 1 footer note)
   - pest_disease:     300–380 words (2–3 stage groups, each with a table + brief notes)
5. Fallback: If no state-specific data exists for a section, use All-India ICAR data.
   Never say "I don't know". Always produce a useful agronomic response with real numbers.
6. Use concrete numbers (kg/ha, DAS, mm, %) from official recommendations.
   Avoid vague phrases like "apply as needed" or "consult an expert".
7. Adjust advice to the given soil type:
   - Black/clay soil → fewer irrigations, watch waterlogging, higher K retention
   - Sandy loam → more frequent irrigation, higher organic matter needs
   - Loam → standard recommendations
   - Laterite → micronutrient-deficient, emphasize Zn/Fe/B supplementation
"""

# ---------------------------------------------------------------------------
# USER PROMPT TEMPLATE
# Edit this to change what context is given per request.
# Variables injected via .format(): crop, state, soil_type, season, irrigation_type
# ---------------------------------------------------------------------------

PRESOWING_USER_TEMPLATE = """Generate a complete pre-sowing advisory for:

- **Crop:** {crop}
- **State:** {state}
- **Soil type:** {soil_type}
- **Season:** {season}
- **Irrigation type:** {irrigation_type}

Use {state}-specific SAU/KVK/ICAR recommendations wherever available.
If {state}-specific data is unavailable for a section, fall back to All-India ICAR guidelines.
Adjust fertilizer, irrigation, and field prep to the soil type: **{soil_type}**.

Each section value must start with its GFM heading. Use this structure:

sowing_window  → ### 🗓️ Sowing Window
seed_selection → ### 🌱 Seed Selection
field_preparation → ### 🔧 Field Preparation
fertilizer_plan   → ### 🧪 Fertilizer Plan
irrigation        → ### 💧 Irrigation Schedule
weed_management   → ### 🌿 Weed Management
pest_disease      → ### 🪲 Pest & Disease Calendar

Now return the JSON with exactly 7 keys for **{crop}** in **{state}**:
"""
