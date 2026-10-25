# JobKit

A local, evidence-first triage tool for job postings, written in Rust.

You give it a job posting and a short profile of yourself. It tells you whether the posting is
remote, what timezone window and years of experience it asks for, which skills it names, and
whether each skill looks **required**, **optional** or **just mentioned**. Every answer comes with
the exact line of the posting it was found on, so you can check it in seconds.

A small local model (Gemma 3 4B through [Ollama](https://ollama.com)) only **names** skills.
Rust verifies and decides everything else. Nothing leaves your machine.

> The model proposes, Rust verifies.

## How it works

1. **Rules (no model).** Plain regex rules read remote status, minimum years, UTC timezone windows,
   and flag work-authorization or location sentences. Remote has three states: `true`, `false`
   (explicitly not remote, onsite, hybrid or in-person), or unknown.
2. **Model.** Gemma lists the skills it sees in the posting.
3. **Verifier.** A skill is accepted only if it appears in the posting as a whole term (known
   aliases such as Postgres/PostgreSQL are allowed). Rust attaches the line it found as evidence.
   A skill the posting never mentions is rejected as **invented**. The model's own quotes are ignored.
4. **Classifier.** Rust classifies each accepted skill from the posting's structure:
   - **required**: some mention reads as a requirement (requirement wording such as "experience
     with" or "you have", or a requirements-style heading above it);
   - **optional**: every mention sits under a nice-to-have style heading ("Nice to have", "Bonus",
     "Preferred", "Good to have:") or its own sentence says "a plus" or "preferred but not required";
   - **mentioned**: it appears only in company prose or stack lists with no requirement wording
     (strict mode only; see below).
5. **Verdict.** `Skip` if the timezone window excludes you or the posting says onsite, hybrid or
   explicitly not remote. `Review` if it contains location or work-authorization sentences, never says
   remote or onsite, or no skills could be extracted. Otherwise `Apply` if at least 70% of the
   required skills match your profile, `Stretch` at 40% or more, and `Skip` below that. A years gap
   is shown in the reasons but doesn't change the verdict.

## Requirements

- Rust (stable) and Cargo
- [Ollama](https://ollama.com) running locally, with a model pulled:

```bash
ollama pull gemma3:4b
```

Developed and tested on WSL (Ubuntu). The model runs locally; CPU works but is slow.

## Quick start

```bash
git clone <this repo>
cd jobkit
# edit profile.toml first (see below)

# rules only: instant, no model
cargo run --release -- analyze postings/example-metalbear.txt --rules-only

# with the model
cargo run --release -- analyze postings/example-metalbear.txt
```

Example output (the example posting is synthetic):

```text
[paste your real output here]
```

## Profile

`profile.toml` describes you:

```toml
years_experience = 3.0   # your total years, compared with the posting's minimum
tz_offset = 1            # hours from UTC (Nigeria is +1)
skills = ["rust", "tokio", "postgres", "docker", "python"]   # matched case-insensitively, aliases allowed
```

## Commands

```bash
jobkit analyze <posting.txt> [--rules-only] [--loose] [--profile profile.toml]
jobkit eval <dir>            [--rules-only] [--loose] [--verbose] [--profile profile.toml]
```

| Flag | Meaning |
|---|---|
| `--rules-only` | Skip the model. Only remote, years, timezone and flags. |
| `--loose` | Treat "mentioned" skills as required. A skill is then required unless **every** mention is optional. Default is strict. |
| `--verbose` | With `eval`: per posting, list extra skills, missed skills, optional and mentioned items, and rejected items with the reason. |
| `--profile` | Path to a profile file (default `profile.toml`). |

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `JOBKIT_MODEL` | `gemma3:4b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama address |
| `JOBKIT_REFRESH` | unset | If set, ignore cached model output and call the model again |

### Cache

Model responses are cached in `.jobkit-cache/`, keyed by the prompt, model and posting text.
Re-scoring after a rule or label change takes seconds, and results are reproducible between runs
(local models are not fully deterministic, even at temperature 0). Delete the folder or set
`JOBKIT_REFRESH=1` to start fresh.

## Evaluating against your own labels

`eval` compares the tool with hand labels. For every `name.txt` in a folder, put a label at
`<dir>/labels/name.json` (or next to the posting as `name.json`):

```json
{"remote": true, "min_years": 3, "tz": [-5, 4], "required_skills": ["rust", "kubernetes"]}
```

- `remote`: `true`, `false`, or `null` if the posting doesn't say.
- `min_years`: a number, or `null` if the posting states no requirement.
- `tz`: `[low, high]` UTC offsets if the posting gives a window like "UTC-5 to UTC+4", otherwise `null`.
- `required_skills`: skills the posting lists and does **not** mark as nice-to-have, good-to-have,
  preferred or a plus.

Label from the posting text only, never from the tool's output. `eval` warns when a labelled skill
doesn't appear anywhere in its posting, which catches typos and concept phrases ("Cloud Platforms")
that the posting never says.

```bash
cargo run --release -- eval postings --verbose
cargo run --release -- eval postings --loose
```

It prints accuracy for remote, years and timezone, and precision and recall for skills.

## Results

Hand-labelled postings, labelled by one person (the author):

| | 9 postings used while building (incl. 1 synthetic) | 8 held-out postings |
|---|---|---|
| remote | 100% (from 78% on the first run) | 62%, then 75% after fixes |
| min years | 100% (from 33%) | 75%, then 100% after fixes |
| timezone | 100% (from 11%) | 100% |
| skills (loose): precision / recall | 0.51 / 0.88 | 0.32 / 0.37 |
| skills (strict): precision / recall | 0.59 / 0.45 | 0.33 / 0.27 |
| invented skills named by the model | 0 of 73 | 1 of 69 (rejected by the verifier) |

The "after fixes" numbers are no longer held-out, because the rules were patched after those failures
were seen. Timing on the author's machine: about 0.08 s per posting for rules only, and 43 to 143 s
per posting with the model.

## Project layout

```text
src/main.rs            everything: rules, verifier, classifier, model call, cache, eval
profile.toml           your profile
postings/              postings to analyse (not committed; see below)
postings/labels/       hand labels
holdout/               a second set, labelled once and run once
.jobkit-cache/         cached model output (generated)
```

Run the tests with `cargo test`.

## Limitations

- Evaluated on 17 postings (16 real, 1 synthetic), all labelled by one person.
- Gemma 3 4B under-extracts long skill lists, and the tool matches terms, not concepts like
  "distributed systems".
- "Mentioned in the posting" is not "required". Stack lists with no requirement wording are
  ambiguous, which is what strict and loose modes expose.
- Boilerplate can trip flags. A pay-transparency sentence containing "US-based" raised a location flag.
- Years are one global number, not per skill.
- The accuracy of the Apply/Stretch/Review/Skip verdict was never measured.

## Privacy and data

Job postings are other people's text. This repository doesn't include real postings, only a
synthetic example. `.gitignore` excludes `postings/*.txt`, `holdout/*.txt` and `.jobkit-cache/`.
Add your own postings locally.

## Possible next steps

- Run the rules first and skip the model for postings that are already a hard Skip (about 35% of the
  model time in the held-out set; an estimate).
- Chunk long skill lists before sending them to the model.
- Measure verdict accuracy against decisions you would actually make.
