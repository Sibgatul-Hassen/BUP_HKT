# BUPGridSchedular — LLM-Assisted Campus Energy Optimizer

BUP CSE Fest 2026 · Hackathon · Online Preliminary

GridWise is one HTTP API. It receives a 24-hour campus energy scenario (hourly
demand, solar forecast, grid tariff, battery limits) plus 1–3 natural-language
operator notes, and returns:

1. a machine-checkable interpretation of **every** note, produced by an LLM and
   validated by deterministic guardrails, and
2. the **cheapest valid** 24-hour battery/grid schedule that obeys every
   applicable directive, found by an exact linear program.

| | |
|---|---|
| **Live endpoint** | `https://buphkt-production.up.railway.app` |
| **Docker image** | `imifty/gridwise:v1.0.0` (digest `sha256:61bb3119ad1cb5cffc0ca69c4b606985c1229796240799ec0aa535ae8877eb06`) |
| **LLM** | Google Gemini `gemini-3.5-flash-lite` (backup `gemini-3.1-flash-lite`) via Gemini's OpenAI-compatible API |
| **Solver** | SciPy `linprog` with the HiGHS LP solver |
| **Endpoints** | `GET /health` · `POST /optimize-energy` · (`GET /docs` interactive API page) |

---

## 1. Quickstart (clean machine, local run)

Requirements: **Python 3.11 or 3.12**, `git`, and a Gemini API key
(free at <https://aistudio.google.com/apikey>).

```bash
# 1. Get the code
git clone https://github.com/Sibgatul-Hassen/BUP_HKT.git
cd BUP_HKT

# 2. Create a virtual environment and install pinned dependencies
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure the LLM (copy the template, then put your key in .env)
cp .env.example .env                 # Windows PowerShell: Copy-Item .env.example .env
#    edit .env and set LLM_API_KEY=<your Gemini key>

# 4. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service reads `.env` automatically at startup. In a second terminal:

```bash
# Health check -> {"status":"ok"}
curl http://localhost:8000/health

# Run one public sample (SAMPLE-06: solar reduction + no-charge window + distractor)
curl -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" \
     --data @data/sample_request.json
```

> **Windows PowerShell:** use `curl.exe` instead of `curl` (in PowerShell,
> `curl` is an alias for `Invoke-WebRequest`):
> `curl.exe -X POST http://localhost:8000/optimize-energy -H "Content-Type: application/json" --data "@data/sample_request.json"`

Expected startup log line: `GridWise starting; LLM configured=True (gemini-3.5-flash-lite, gemini-3.1-flash-lite @ generativelanguage.googleapis.com)`.
If it says `LLM configured=False`, the `.env` file is missing or incomplete.

---

## 2. Run the public sample cases (test procedure + expected result)

`scripts/run_samples.py` posts every case in `data/public_samples.json` and
grades the response: HTTP status, `scenario_id` echo, each note's
interpretation (type, `applies`, hours, numeric value within 0.01) against
the reference, an **independent hour-by-hour replay** of the plan against the
reference directives, and cost versus the organizer optimum.

```bash
python -m scripts.run_samples                                          # in-process, no server needed
python -m scripts.run_samples --url http://localhost:8000              # a running local server
python -m scripts.run_samples --url https://buphkt-production.up.railway.app   # the live deployment
```

Expected result (latency varies with the LLM provider):

```
/health -> 200 {"status":"ok"}
PASS SAMPLE-01  cost=38365     ratio=1.0000 1.83s
PASS SAMPLE-02  cost=42885     ratio=1.0000 1.00s
...
PASS SAMPLE-10  cost=41620     ratio=1.0000 1.61s

10/10 passed; p95 latency ...
```

All 10 public cases pass with **exactly the organizer's optimal cost**
(ratio 1.0000). The script exits non-zero if any case fails.

---

## 3. Docker fallback image

The image contains **no secrets**; the API key is passed at run time. It
exposes port **8000** and binds `0.0.0.0`.

```bash
docker pull imifty/gridwise:v1.0.0

docker run --rm -p 8000:8000 \
  -e LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  -e LLM_MODEL=gemini-3.5-flash-lite \
  -e LLM_FALLBACK_MODELS=gemini-3.1-flash-lite \
  -e LLM_API_KEY=<your Gemini key> \
  imifty/gridwise:v1.0.0
```

Or, from a clone with a filled-in `.env`:

```bash
docker run --rm -p 8000:8000 --env-file .env imifty/gridwise:v1.0.0
```

Then `curl http://localhost:8000/health` → `{"status":"ok"}` (ready in well
under 60 s) and `python -m scripts.run_samples --url http://localhost:8000`.

Build it yourself: `docker build -t gridwise .`

---

## 4. Configuration

All configuration is environment variables. Switching LLM provider is a
configuration change, never a code change — any OpenAI-compatible
`/chat/completions` endpoint works (Gemini, Groq, OpenRouter, Cerebras, …).

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `LLM_BASE_URL` | yes | – | OpenAI-compatible base URL, e.g. `https://generativelanguage.googleapis.com/v1beta/openai` |
| `LLM_MODEL` | yes | – | Primary model, e.g. `gemini-3.5-flash-lite` |
| `LLM_API_KEY` | yes | – | Provider API key (secret — never commit it) |
| `LLM_FALLBACK_MODELS` | no | none | Comma-separated backup models tried when the primary is rate-limited, slow or returns unusable output |
| `LLM_TIMEOUT_SECONDS` | no | `8` | Timeout of a single model attempt |
| `LLM_TOTAL_BUDGET_SECONDS` | no | `20` | Total time allowed for all model attempts per request (judge timeout is 30 s) |
| `PORT` | no | `8000` | HTTP port (hosting platforms such as Railway set this automatically) |
| `LOG_LEVEL` | no | `INFO` | Python logging level |

`.env.example` lists the names with placeholder values.

---

## 5. Architecture: LLM → guardrails → optimizer → replay

```
 POST /optimize-energy
        │
        ▼
 ┌──────────────────┐  malformed / wrong shape → 400; inconsistent battery → 422
 │ Request schema   │  (exactly 24 unique hours, 1–3 non-empty notes, finite ≥ 0 numbers)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐  ONE batched chat call for all notes (temperature 0, JSON mode).
 │ LLM reading      │  Model reports only what it READ: directive type, clock-hour
 │ app/llm.py       │  windows, the raw quantity and its kind ("80% reduction",
 └────────┬─────────┘  "50% of capacity", "155 kWh"). It does no arithmetic.
          ▼
 ┌──────────────────┐  Deterministic code does all the math and validation:
 │ Guardrails       │  windows → hours (start-inclusive, end-exclusive, midnight wrap),
 │ app/guardrails.py│  "80% reduction" → factor 0.2, "50% of 200 kWh" → 100 kWh,
 └────────┬─────────┘  allowed types only, hours unique ascending 0–23, factor in [0,1],
          │            0 ≤ reserve ≤ capacity, grid cap ≥ 0, applies/no_op semantics,
          │            exactly one entry per note in note_index order.
          ▼
 ┌──────────────────┐  Exact linear program (72 variables: charge, discharge and
 │ Optimizer        │  solar-used per hour). Directives become hard constraints.
 │ app/optimizer.py │  Minimises Σ tariff[h] · grid[h]. Solved by HiGHS.
 └────────┬─────────┘
          ▼
 ┌──────────────────┐  Independent hour-by-hour replay of the finished plan — the same
 │ Replay check     │  checks the judge runs. Totals are recomputed from hourly_plan.
 │ app/replay.py    │
 └────────┬─────────┘
          ▼
   JSON response: scenario_id, directive_interpretation, hourly_plan,
   total_grid_kwh, total_cost_bdt, peak_grid_kwh, plan_summary
```

### Role of the LLM

The LLM is the interpreter of `operator_notes`. Its structured reading is what
becomes `directive_interpretation` and the optimizer's constraints. The prompt
(`SYSTEM_PROMPT` in `app/llm.py`) defines the six directive types, the time
conventions and the quantity kinds, and tells the model to treat note text as
data — instructions embedded in a note are never followed.

**Why the model does no arithmetic:** hour lists and factor/kWh conversions are
exactly where language models slip ("80% reduction" vs "drops to 20%",
"1 PM to 3 PM" → `[13, 14]`, "half of capacity"). Doing them in code makes
them exact and testable.

### Guardrails and safe failure

- LLM output is untrusted until `normalize()` in `app/guardrails.py` accepts it.
- If a single note's reading is rejected, only that note falls back (see below);
  the others keep their LLM reading.
- Nothing can invent a directive type: anything outside the six allowed types
  is rejected; the last resort is `no_op`, never a made-up constraint.
- `validate_final()` re-checks the complete interpretation list against the
  Problem Statement contract before the optimizer sees it.

### Optimizer

Variables per hour `h`: charge `c[h]`, discharge `d[h]`, solar used `s[h]`.
Grid import is derived: `g[h] = demand[h] + c[h] − s[h] − d[h]`.

- `0 ≤ s[h] ≤ effective_solar[h]` (after `solar_reduction`; surplus is curtailed)
- `0 ≤ c[h] ≤ max_charge` (0 inside a `no_charge_window`)
- `0 ≤ d[h] ≤ max_discharge` (0 inside a `no_discharge_window`)
- `g[h] ≥ 0`, and `g[h] ≤ max_grid_kwh` inside a `max_grid_window`
- battery energy after each hour between the (possibly raised) minimum and capacity
- `Σ c − Σ d = 0` (end-of-day battery neutrality)

Two post-processing steps: charge and discharge are netted per hour (so each
hour has one `battery_action`), and battery energy is re-simulated from the
netted values so `battery_energy_after_kwh` is exactly consistent.

### Reliability

| Situation | Behaviour |
|---|---|
| Malformed JSON / wrong structure | HTTP 400 with a short message, no stack trace |
| Battery min/initial/capacity inconsistent | HTTP 422 |
| Model timeout, rate limit (429), 5xx, invalid JSON | retry on the next model in `LLM_FALLBACK_MODELS`, within the time budget |
| Bad credentials / no quota (401/402/403) | no pointless retries |
| Provider completely unavailable | **outage fallback reader** (below) keeps the service answering |
| Interpreted directives cannot all be satisfied | keep the largest satisfiable subset (judge scenarios are feasible, so this only protects against a misread) |
| Unexpected exception | HTTP 500 `{"error": "internal error"}` |
| Repeated notes | validated LLM readings are cached by note text; repeats skip the network |

**Outage fallback reader (`app/fallback_parser.py`).** A conservative
rule-based reader used **only** when the LLM provider is unreachable or a
note's LLM reading fails the guardrails. It emits the same reading format and
passes through the same guardrails, and says `no_op` when unsure. It is a
resilience measure, **not** the interpreter: in normal operation every note is
read by the LLM. Responses produced by it carry an explanation starting with
`Fallback reader:`.

---

## 6. API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /optimize-energy`

Request (abridged — full example in `data/sample_request.json`):

```json
{
  "scenario_id": "SAMPLE-06",
  "operator_notes": [
    "Cloud cover during panel inspection will leave about half of the forecast solar output from 10 AM until noon.",
    "The charging circuit will be unavailable from 2 PM until 4 PM.",
    "The library is extending book-return hours next week."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
    "... 24 entries, hours 0-23 ..."
  ],
  "battery": {
    "capacity_kwh": 220, "initial_energy_kwh": 100, "minimum_energy_kwh": 35,
    "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
  }
}
```

Response (abridged):

```json
{
  "scenario_id": "SAMPLE-06",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [10, 11], "factor": 0.5},
     "explanation": "Solar output is reduced to half of forecast between 10 AM and noon."},
    {"note_index": 1, "applies": true, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]},
     "explanation": "The charging circuit is unavailable from 2 PM to 4 PM."},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "The note refers to library hours next week and does not affect today's energy schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 105.0, "solar_used_kwh": 0.0, "battery_action": "charge",
     "battery_kwh": 20.0, "battery_energy_after_kwh": 120.0},
    "... 24 entries ..."
  ],
  "total_grid_kwh": 2395.0,
  "total_cost_bdt": 34090.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Applies 2 operator directive(s) as hard constraints (no charge window, solar reduction); ignores 1 unrelated note(s). ..."
}
```

Status codes: `200` success · `400` malformed JSON or structurally invalid
request · `422` well-formed but inconsistent battery values · `500`
controlled internal error.

---

## 7. Project layout

```
app/
  main.py             FastAPI app: endpoints, error handling, scheduling flow
  schemas.py          request validation (400 / 422)
  llm.py              LLM client: prompt, batched call, retries, model rotation, cache
  guardrails.py       deterministic validation + all unit/hour arithmetic
  interpret.py        per-note trust order: LLM -> outage fallback -> no_op
  fallback_parser.py  outage-only rule-based reader
  optimizer.py        linear program (SciPy / HiGHS)
  replay.py           independent validity check of every plan
scripts/run_samples.py  public-sample grader (local or remote)
data/public_samples.json  organizer public cases (test fixture only)
data/sample_request.json  one request body for curl
Dockerfile, requirements.txt, .env.example
```

---

## 8. Secret handling

- The API key is supplied only through environment variables (`.env` locally,
  the platform's variable settings in deployment, `-e`/`--env-file` for Docker).
- `.env` is in `.gitignore` and `.dockerignore`; the Docker image contains no
  key or `.env` file. `.env.example` carries variable names only.
- The key is sent only in the `Authorization` header to the configured
  provider. It is never logged and never returned in responses; logs show only
  the model name and provider host. Error responses contain no stack traces.

---

## 9. Known limitations

- **Hosted-model dependency.** Interpretation quality and latency depend on the
  LLM provider. Free-tier keys have per-minute limits; under heavy bursts some
  requests may use the outage fallback reader, which handles common phrasings
  but is less robust to unusual paraphrases than the LLM.
- **Latency tail.** Typical requests take ~1–2 s; an occasional slow model
  response can push a request to ~7–8 s (always well inside the 30 s limit).
- **One directive per note.** Each note maps to exactly one directive type, as
  the Problem Statement guarantees; a single note describing two different
  constraints would keep only one.
- **In-memory cache.** The interpretation cache is per process and is lost on
  restart (it only saves latency; correctness does not depend on it).
- **Scope.** No grid export, no battery round-trip losses, 24 hourly steps —
  exactly the GridWise model in the Problem Statement.

---

## 10. Dependencies and credits

| Dependency | Use |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) 0.115 · [Uvicorn](https://www.uvicorn.org/) 0.34 | HTTP API and server |
| [Pydantic](https://docs.pydantic.dev/) 2.10 | request validation |
| [HTTPX](https://www.python-httpx.org/) 0.28 | async client for the LLM API |
| [NumPy](https://numpy.org/) 2.2 · [SciPy](https://scipy.org/) 1.15 ([HiGHS](https://highs.dev/) solver) | linear program |
| [python-dotenv](https://github.com/theskumar/python-dotenv) 1.0 | loads `.env` for local runs |
| [Google Gemini API](https://ai.google.dev/) | LLM for operator-note interpretation |
| [Railway](https://railway.com/) · [Docker Hub](https://hub.docker.com/) | hosting and image registry |

AI coding assistance (Claude Code) was used during development, as permitted
by the rulebook; the architecture and design decisions are the team's own.
All scenario data is the synthetic organizer data.
