import os
import json
import datetime
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# === EDIT THIS: put your input file path here (absolute or relative) ===
EXCEL_PATH = r""   # <- set your Excel filename here
# === end edit ==========================================================

# --- Load .env ---------------------------------------------------------------
load_dotenv(dotenv_path=".env")

def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Missing environment variable: {var}. Please set it in your .env file.")
    return val

# --- Azure & Google env vars (must exist in .env) --------------------------
AZURE_API_KEY = _require_env("AZURE_API_KEY")
AZURE_API_BASE = _require_env("AZURE_API_BASE")
AZURE_API_VERSION = _require_env("AZURE_API_VERSION")
AZURE_DEPLOYMENT_NAME = _require_env("AZURE_DEPLOYMENT_NAME")
GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")

# --- Quiet telemetry --------------------------------------------------------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", os.getenv("CREWAI_DISABLE_TELEMETRY", "true"))
os.environ.setdefault("OTEL_SDK_DISABLED", os.getenv("OTEL_SDK_DISABLED", "true"))
os.environ.setdefault("OPENAI_API_KEY", "")

# --- Route Azure settings for CrewAI/LLM -----------------------------------
os.environ["AZURE_API_KEY"] = AZURE_API_KEY
os.environ["AZURE_API_BASE"] = AZURE_API_BASE
os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION

# --- Excel deps -------------------------------------------------------------
import pandas as pd  # pip install pandas openpyxl

# --- CrewAI imports ---------------------------------------------------------
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import BaseTool
from pydantic import PrivateAttr

# ----------------------
# GoogleGeminiSearchTool (Pydantic + model_post_init)
# ----------------------
class GoogleGeminiSearchTool(BaseTool):
    name: str = "GoogleGeminiSearchTool"
    description: str = "Search the web via Google's Gemini grounded search and return a JSON list of result URLs and titles."
    model: str = "gemini-2.0-flash"

    _client: object = PrivateAttr(default=None)
    _genai: object = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY in your environment/.env")
        os.environ["GOOGLE_API_KEY"] = key
        try:
            from google import genai as genai
            self._genai = genai
            self._client = genai.Client()
        except Exception as e:
            raise RuntimeError("Install google-genai: pip install google-genai") from e

    def _extract_urls(self, response) -> List[Dict[str, str]]:
        urls = []
        try:
            cand = response.candidates[0]
            gm = getattr(cand, "grounding_metadata", None)
            if gm and getattr(gm, "grounding_chunks", None):
                for ch in gm.grounding_chunks:
                    web = getattr(ch, "web", None)
                    if web:
                        uri = getattr(web, "uri", None) or getattr(web, "url", None)
                        title = getattr(web, "title", None) or ""
                        if uri:
                            urls.append({"url": uri, "title": title})
        except Exception:
            pass
        seen, dedup = set(), []
        for it in urls:
            u = it.get("url")
            if u and u not in seen:
                seen.add(u)
                dedup.append(it)
        return dedup

    def _run(self, query: str) -> str:
        if self._client is None:
            raise RuntimeError("GoogleGeminiSearchTool not initialized: _client is None")
        resp = self._client.models.generate_content(
            model=self.model,
            contents=query,
            config={"tools": [{"google_search": {}}]},
        )
        results = self._extract_urls(resp)
        if not results:
            return json.dumps([{"url": "", "title": "", "why": "No URLs grounded by Google search"}], ensure_ascii=False)
        annotated = [{"url": r["url"], "title": r.get("title", ""), "why": "Relevant per Google search"} for r in results]
        return json.dumps(annotated[:8], ensure_ascii=False)

# --- LLM (Azure) -------------------------------------------------------------
llm = LLM(
    model=f"azure/{AZURE_DEPLOYMENT_NAME}",
    temperature=0.2,
    max_tokens=1200,
)

# --- Tools --------------------------------------------------------------
search_tool = GoogleGeminiSearchTool()
scrape_tool = ScrapeWebsiteTool()

# --- ExcelReaderTool ------------------------------------------------------
class ExcelReaderTool(BaseTool):
    name: str = "ExcelReaderTool"
    description: str = "Reads an Excel file and returns rows with Component and Company."

    def _run(self, path: str) -> str:
        if not os.path.exists(path):
            raise RuntimeError(f"Excel file not found: {path}")
        df = pd.read_excel(path, engine="openpyxl")
        cols_map = {c.strip(): c for c in df.columns}
        comp_col = None
        company_col = None
        for k in cols_map:
            lk = k.lower()
            if 'component' == lk or 'component' in lk:
                comp_col = cols_map[k]
            if 'company' == lk or 'company' in lk:
                company_col = cols_map[k]
        if comp_col is None:
            comp_col = df.columns[0]
        rows = []
        for idx, row in df.iterrows():
            comp = row.get(comp_col, "")
            comp = "" if pd.isna(comp) else str(comp).strip().replace("\n", " ")
            company = row.get(company_col, "") if company_col is not None else ""
            company = "" if pd.isna(company) else str(company).strip()
            rows.append({"row_index": int(idx), "Component": comp, "Company": company})
        return json.dumps(rows, ensure_ascii=False)

# --- Power Normalizer Tool -----------------------------------------
class PowerNormalizerTool(BaseTool):
    """Takes the pipeline's parsed object (dict or string) and returns normalized power in watts,
    heat (BHU), and a best source URL if available.

    Assumptions:
      - If VA found but no power factor (PF), assumes PF = 1.0 (i.e., VA -> W one-to-one).
      - Recognizes units: W, kW, kVA, VA, MW, HP (approx conversion for HP -> W using 745.699872).
    """
    name: str = "PowerNormalizerTool"
    description: str = "Normalize power strings (including VA) to watts, compute heat, and extract source URL."

    def _run(self, parsed_json_str: str) -> str:
        # parsed_json_str can be a JSON string or plain text
        try:
            parsed = json.loads(parsed_json_str)
        except Exception:
            parsed = parsed_json_str

        # Convert parsed object into a textual blob to search for numbers & units
        text_blob = ''
        if isinstance(parsed, dict):
            # look for common fields
            if 'answer_bullets' in parsed and isinstance(parsed['answer_bullets'], list):
                text_blob = ' '.join(parsed['answer_bullets'])
            elif 'raw_output' in parsed and isinstance(parsed['raw_output'], str):
                text_blob = parsed['raw_output']
            else:
                # fallback: stringify
                text_blob = json.dumps(parsed)
        elif isinstance(parsed, list):
            text_blob = ' '.join([json.dumps(p) for p in parsed])
        else:
            text_blob = str(parsed)

        # 1) try to extract a best URL
        url = self._extract_best_url(parsed, text_blob)

        # 2) find PF if available (pf=0.8 etc.)
        pf = self._extract_power_factor(text_blob)

        # 3) find numeric power (consider VA, kVA, W, kW, MW, HP)
        watts = self._extract_power_to_watts(text_blob, pf)

        # 4) prepare return object
        if watts is not None:
            # keep integer if near-int
            if abs(watts - round(watts)) < 1e-9:
                power_text = f"{int(round(watts))} W"
            else:
                power_text = f"{round(watts, 3)} W"
            bhu = watts * 3.414
            bhu_val_rounded = round(bhu, 2) if bhu >= 1 else round(bhu, 3)
            heat_text = f"{bhu_val_rounded} BHU"
        else:
            power_text = ''
            heat_text = ''

        return json.dumps({
            'watts': watts,
            'power_text': power_text,
            'heat_text': heat_text,
            'source_url': url,
            'power_factor_used': pf,
        }, ensure_ascii=False)

    def _extract_best_url(self, parsed: Any, text_blob: str) -> Optional[str]:
        # 1) if parsed is a dict and has 'sources' key that's a list, prefer the first
        if isinstance(parsed, dict):
            for k in ('sources', 'source', 'urls', 'references'):
                if k in parsed:
                    val = parsed[k]
                    if isinstance(val, list) and len(val) > 0:
                        # items might be strings or dicts with 'url'
                        first = val[0]
                        if isinstance(first, str) and first.startswith('http'):
                            return first
                        if isinstance(first, dict):
                            if 'url' in first and isinstance(first['url'], str):
                                return first['url']
                            # sometimes item is {'title':..., 'url':...}
        # 2) search for any http(s) in text_blob
        m = re.search(r'https?://[^\s\)\]\"\']+', text_blob)
        if m:
            return m.group(0)
        return None

    def _extract_power_factor(self, text: str) -> float:
        # find patterns like 'pf 0.8' or 'power factor 0.85'
        m = re.search(r'pf\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r'power factor\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0 < val <= 1.5:
                    return val
            except Exception:
                pass
        return 1.0

    def _extract_power_to_watts(self, text: str, pf: float = 1.0) -> Optional[float]:
        s = text.lower()
        # normalize separators
        s = s.replace(',', ' ')

        # Pattern: number + optional space + unit
        patterns = [
            (r'([0-9]+(?:\.[0-9]+)?)\s*(kw)\b', 1000.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(kva)\b', 1000.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(va)\b', 1.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(w)\b', 1.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(mw)\b', 1000000.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(mwatt)\b', 1000000.0),
            (r'([0-9]+(?:\.[0-9]+)?)\s*(hp)\b', 745.699872),
            # allow units attached without space like '596va' or '0.48kw'
            (r'([0-9]+(?:\.[0-9]+)?)(kw)\b', 1000.0),
            (r'([0-9]+(?:\.[0-9]+)?)(kva)\b', 1000.0),
            (r'([0-9]+(?:\.[0-9]+)?)(va)\b', 1.0),
            (r'([0-9]+(?:\.[0-9]+)?)(w)\b', 1.0),
        ]

        for pat, multiplier in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                try:
                    num = float(m.group(1))
                    unit = m.group(2)
                    # If unit was VA or KVA, convert to W using PF
                    if unit.lower() in ('va', 'kva'):
                        watts = num * multiplier * float(pf)
                    else:
                        watts = num * multiplier
                    return watts
                except Exception:
                    continue

        # If nothing matched, try to find plain numbers followed by word 'va' or 'w' in verbose text
        m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:volt-amp|volt amp|voltamp)\b', s)
        if m2:
            try:
                return float(m2.group(1)) * float(pf)
            except Exception:
                pass

        return None

# --- Agents ------------------------------------------------------------------
researcher = Agent(
    role="Web Researcher",
    goal="Find trustworthy, recent sources that directly answer the user's question.",
    backstory=(
        "Expert at search tactics, query reformulation, and source quality checks. "
        "Prefers primary docs and reputable outlets."
    ),
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

extractor = Agent(
    role="Content Extractor",
    goal="Open links and extract concise, well-attributed facts that answer the question.",
    backstory=(
        "Careful reader that pulls key facts, numbers, dates, and short quotes "
        "with clear source URLs."
    ),
    tools=[scrape_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

synthesizer = Agent(
    role="Evidence-based Summarizer",
    goal="Synthesize a short, accurate answer with citations to the scraped sources.",
    backstory="Merges extracted points, deduplicates overlaps, and flags contradictions.",
    tools=[],
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# --- Normalizer agent ------------------------------------------------
power_norm_tool = PowerNormalizerTool()
normalizer = Agent(
    role="Power Normalizer",
    goal="Given the synthesizer's parsed JSON, extract numeric power, normalize to W, compute heat, and provide a best source URL.",
    backstory="Specialised at unit normalization and engineering conversions.",
    tools=[power_norm_tool],
    allow_delegation=False,
    verbose=False,
    llm=llm,
)

# --- Tasks -------------------------------------------------------------------
def build_tasks_for_question(question: str) -> List[Task]:
    search_task = Task(
        description=(
            f'Question: "{question}"\n'
            "1) Use the GoogleGeminiSearchTool to find 5–8 high-quality, recent sources.\n"
            "2) Return a JSON list with objects: {'url': str, 'title': str, 'why': str}.\n"
            "Avoid duplicates and low-quality spam."
        ),
        agent=researcher,
        expected_output="JSON list of candidate sources."
    )

    scrape_task = Task(
        description=(
            f'Question: "{question}"\n'
            "Read the JSON list from the previous step. For each URL:\n"
            " - Use ScrapeWebsiteTool to fetch content.\n"
            "Extract compact facts that answer the question with fields:\n"
            "  {'url': str, 'title': str|null, 'key_facts': list, 'numbers_or_dates': list, 'quotes': list}\n"
            "Return a JSON list of these per-URL objects."
        ),
        agent=extractor,
        expected_output="JSON list of per-URL extractions.",
        depends_on=[search_task],
    )

    synthesize_task = Task(
        description=(
            f'Combine all extracted JSON into a final, concise answer for: \"{question}\"' 
            " - 4–8 bullet points, strictly evidence-based "
            " - include a 'Sources' section listing the top 3–6 URLs used "
            "Return a JSON object: {'answer_bullets': list[str], 'sources': list[str]}"
        ),
        agent=synthesizer,
        expected_output="Final JSON with answer_bullets + sources.",
        depends_on=[scrape_task],
    )

    return [search_task, scrape_task, synthesize_task]


def answer_questions(questions: List[str]) -> Dict[str, Any]:
    results = {}
    for q in questions:
        tasks = build_tasks_for_question(q)
        crew = Crew(
            agents=[researcher, extractor, synthesizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        final_text = crew.kickoff()
        try:
            parsed = json.loads(str(final_text))
        except Exception:
            parsed = {"raw_output": str(final_text)}
        intermediate = {}
        for t in tasks:
            out = getattr(t, "output", None)
            try:
                if hasattr(out, "json_dict") and out.json_dict is not None:
                    intermediate[t.description.splitlines()[0]] = out.json_dict
                elif hasattr(out, "raw") and out.raw:
                    intermediate[t.description.splitlines()[0]] = json.loads(out.raw)
                else:
                    intermediate[t.description.splitlines()[0]] = str(out)
            except Exception:
                intermediate[t.description.splitlines()[0]] = str(out)
        results[q] = {"final": parsed, "intermediate": intermediate}
    return results

# --- Power extraction + normalization utilities ----------------------------
def extract_numeric_power_watts_from_text(text: str) -> Optional[float]:
    """
    Returns numeric power in W if found, otherwise None.
    Accepts patterns like:
      '480 W', '480W', '0.48 kW', '480 watts', '480 watt'
      Also handles ranges by taking the first numeric value.
    """
    if not text:
        return None
    s = text.lower()
    # find first numeric + unit match
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s?(kw|kW|w|mw|watts|watt)\b', s, flags=re.IGNORECASE)
    if not m:
        # try numeric alone (fallback) but prefer explicit units
        m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s?(w)\b', s, flags=re.IGNORECASE)
        if not m2:
            return None
        m = m2
    try:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("kw", "kW", "kw"):
            watts = num * 1000.0
        elif unit in ("mw",):
            watts = num / 1000.0
        else:
            # 'w' or 'watt' or 'watts' or unknown -> treat as W
            watts = num
        return watts
    except Exception:
        return None
    
# --- MAIN (no CLI args) -----------------------------------------------------
if __name__ == "__main__":
    # Input
    input_path = EXCEL_PATH

    if not os.path.exists(input_path):
        raise SystemExit(f"Input Excel not found: {input_path}")

    # 1) Read rows via ExcelReaderTool
    reader = ExcelReaderTool()
    rows_json = reader._run(input_path)
    rows = json.loads(rows_json)

    # 2) Load dataframe to write back
    df = pd.read_excel(input_path, engine="openpyxl")

    # Ensure Power, Heat and Source columns exist
    if 'Power' not in df.columns:
        df['Power'] = ""
    if 'Heat' not in df.columns:
        df['Heat'] = ""
    if 'Source' not in df.columns:
        df['Source'] = ""

    results_per_row = {}

    for r in rows:
        idx = r.get("row_index")
        comp = r.get("Component", "").strip()
        company = r.get("Company", "").strip()
        if not comp:
            results_per_row[idx] = {"error": "No component"}
            continue

        # Build focused query
        query = f"Find the input power consumption (value + units) for the product '{comp}' from {company}. Give concrete numeric value(s) with units and list the best source."

        try:
            out = answer_questions([query])
            parsed = out.get(query, {}).get("final", {})
        except Exception as e:
            parsed = {"raw_output": f"Error running pipeline: {e}"}

        # Run normalizer tool (NEW) to detect VA/kVA/kW and extract source
        try:
            norm_raw = power_norm_tool._run(json.dumps(parsed, ensure_ascii=False))
            norm = json.loads(norm_raw)
        except Exception as e:
            norm = {'watts': None, 'power_text': '', 'heat_text': '', 'source_url': None, 'power_factor_used': 1.0}

        watts = norm.get('watts')
        power_text = norm.get('power_text', '')
        heat_text = norm.get('heat_text', '')
        source_url = norm.get('source_url')

        # Fallback: if normalizer didn't find numeric watts, try old text parser on synthesizer text
        if watts is None:
            text_for_parse = ''
            # reuse previous heuristic
            if isinstance(parsed, dict):
                ab = parsed.get('answer_bullets')
                if isinstance(ab, list) and len(ab) > 0:
                    text_for_parse = ' '.join(ab)
                else:
                    ro = parsed.get('raw_output')
                    if isinstance(ro, str):
                        text_for_parse = ro
                    else:
                        text_for_parse = json.dumps(parsed)
            elif isinstance(parsed, str):
                text_for_parse = parsed
            else:
                text_for_parse = str(parsed)

            watts = extract_numeric_power_watts_from_text(text_for_parse)
            if watts is not None:
                # compute texts same as before
                if abs(watts - round(watts)) < 1e-9:
                    power_text = f"{int(round(watts))} W"
                else:
                    power_text = f"{round(watts, 3)} W"
                bhu_val = watts * 3.414
                bhu_val_rounded = round(bhu_val, 2) if bhu_val >= 1 else round(bhu_val, 3)
                heat_text = f"{bhu_val_rounded} BHU"

        # If still no watts, fallback to plain text from parsed
        if watts is None and (not power_text):
            fallback_text = ''
            if isinstance(parsed, dict):
                ab = parsed.get('answer_bullets')
                if isinstance(ab, list) and len(ab) > 0:
                    fallback_text = ' '.join(ab)
                else:
                    ro = parsed.get('raw_output')
                    fallback_text = ro if isinstance(ro, str) else json.dumps(parsed)
            elif isinstance(parsed, str):
                fallback_text = parsed
            else:
                fallback_text = str(parsed)
            fallback_text_short = fallback_text.strip()
            if len(fallback_text_short) > 200:
                fallback_text_short = fallback_text_short[:197] + '...'
            power_text = fallback_text_short
            heat_text = ''

        # Write back into dataframe (use index mapping)
        try:
            df.at[int(idx), 'Power'] = power_text
            df.at[int(idx), 'Heat'] = heat_text
            df.at[int(idx), 'Source'] = source_url or ''
        except Exception:
            # fallback: match by component string in first column
            try:
                mask = df[df.iloc[:,0].astype(str).str.strip() == comp].index
                if len(mask) > 0:
                    df.at[mask[0], 'Power'] = power_text
                    df.at[mask[0], 'Heat'] = heat_text
                    df.at[mask[0], 'Source'] = source_url or ''
            except Exception:
                pass

        results_per_row[idx] = {"component": comp, "company": company, "power_watts": watts, "power_text": power_text, "heat_text": heat_text, "source_url": source_url, "parsed": parsed}

    # 4) Save modified Excel & JSON into same folder as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_excel = os.path.join(script_dir, f"crew_ai_filled_{ts}.xlsx")
    out_json = os.path.join(script_dir, f"crew_ai_results_{ts}.json")

    df.to_excel(out_excel, index=False, engine="openpyxl")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(results_per_row, jf, indent=2, ensure_ascii=False)

    print(f"Saved: {out_excel}")
    print(f"Saved: {out_json}")
