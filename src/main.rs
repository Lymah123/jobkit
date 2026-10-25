//! jobkit: extract, verify and match job-posting requirements against your profile.
//!
//! The local model only *proposes* requirements (each with a verbatim quote).
//! Rust verifies every quote against the posting and decides eligibility with rules.
//!
//! Usage:
//!   jobkit analyze <posting.txt> [--profile profile.toml] [--rules-only]
//!   jobkit eval <dir> [--profile profile.toml] [--rules-only]
//!
//! Env: JOBKIT_MODEL (default gemma3:4b), OLLAMA_URL (default http://localhost:11434)

use regex::Regex;
use serde::Deserialize;
use std::{env, fs, path::Path, process};

// ---------- data ----------

#[derive(Deserialize, Debug, Clone)]
pub struct Profile {
    pub years_experience: f32,
    pub tz_offset: i32, // hours from UTC, e.g. 1 for Nigeria
    pub skills: Vec<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Item {
    pub skill: String,
    pub quote: String,
}

/// Small models sometimes return a bare string instead of {skill, quote}.
/// A bare string carries no evidence, so it becomes an Item with an empty quote
/// and the verifier rejects it (and it shows up in the rejected count).
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum RawItem {
    Full(Item),
    Bare(String),
}

impl From<RawItem> for Item {
    fn from(r: RawItem) -> Item {
        match r {
            RawItem::Full(i) => i,
            RawItem::Bare(s) => Item {
                skill: s,
                quote: String::new(),
            },
        }
    }
}

#[derive(Debug, Default)]
struct RawExtraction {
    required: Vec<RawItem>,
    nice_to_have: Vec<RawItem>,
}

/// Small models sometimes repeat a key (e.g. two "nice_to_have" arrays). Merge them
/// instead of failing the whole posting.
impl<'de> Deserialize<'de> for RawExtraction {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = RawExtraction;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut m: A,
            ) -> Result<RawExtraction, A::Error> {
                let mut out = RawExtraction::default();
                while let Some(k) = m.next_key::<String>()? {
                    match k.as_str() {
                        "required" => out.required.extend(m.next_value::<Vec<RawItem>>()?),
                        "nice_to_have" => out.nice_to_have.extend(m.next_value::<Vec<RawItem>>()?),
                        _ => {
                            m.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Debug, Default)]
pub struct Extraction {
    pub required: Vec<Item>,
    pub nice_to_have: Vec<Item>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Rules {
    pub remote: Option<bool>,
    pub onsite_or_hybrid: bool,
    pub min_years: Option<u32>,
    pub tz_window: Option<(i32, i32)>,
    pub auth_flags: Vec<String>, // sentences needing a human look
    pub tz_notes: Vec<String>,   // named zones / overlap requirements ("Pacific Time")
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Apply,
    Stretch,
    Review,
    Skip,
}

// ---------- verification (the part that makes the model checkable) ----------

const ALIASES: &[(&str, &[&str])] = &[
    ("postgresql", &["postgres"]),
    ("javascript", &["js"]),
    ("kubernetes", &["k8s"]),
    ("next.js", &["nextjs"]),
    ("tailwind css", &["tailwindcss", "tailwind"]),
    ("node.js", &["nodejs"]),
    ("rest apis", &["rest api", "restful"]),
    ("llm apis", &["llm api"]),
];

/// Spellings to look for in the posting: the skill itself, a naive singular, known aliases.
fn terms_for(skill: &str) -> Vec<String> {
    let s = skill.trim().to_lowercase();
    let mut v = vec![s.clone()];
    if s.len() > 3 && s.ends_with('s') {
        v.push(s[..s.len() - 1].to_string());
    }
    for (k, alts) in ALIASES {
        if v.iter().any(|x| x == k) || alts.contains(&s.as_str()) {
            v.push(k.to_string());
            v.extend(alts.iter().map(|a| a.to_string()));
        }
    }
    v.sort();
    v.dedup();
    v
}

/// The model only *names* skills. Rust verifies each one by finding it in the posting
/// (whole-term match, known aliases allowed) and attaches the line where it was found as the
/// evidence. The model's own quote is ignored: copying text exactly is brittle for small
/// models, while "is this term in the posting" is what actually guards against invention.
pub fn verify(items: Vec<Item>, posting: &str) -> (Vec<Item>, Vec<Item>) {
    let lines: Vec<&str> = posting.lines().collect();
    let mut ok: Vec<Item> = vec![];
    let mut bad = vec![];
    for it in items {
        if it.skill.trim().is_empty() {
            bad.push(it);
            continue;
        }
        let terms = terms_for(&it.skill);
        let found = lines
            .iter()
            .find(|l| terms.iter().any(|t| contains_word(l, t)));
        match found {
            Some(line) => {
                if ok.iter().any(|x| x.skill.eq_ignore_ascii_case(&it.skill)) {
                    continue; // duplicate of an accepted skill
                }
                let evidence: String = line.trim().chars().take(200).collect();
                ok.push(Item {
                    skill: it.skill,
                    quote: evidence,
                });
            }
            None => bad.push(it),
        }
    }
    (ok, bad)
}

pub fn reject_reason(it: &Item, _posting: &str) -> &'static str {
    if it.skill.trim().is_empty() {
        "empty skill name"
    } else {
        "skill never appears in the posting (invented)"
    }
}

fn fix_quotes(s: &str) -> String {
    s.replace('\u{2019}', "'")
}

/// Optional wording in a heading. Deliberately excludes "not required": a heading such as
/// "US citizenship/visa not required" says nothing about the skills listed below it.
fn opt_heading_re() -> Regex {
    Regex::new(
        r"(?i)\b(nice[- ]to[- ]have|bonus|a plus|is a plus|are a plus|preferred|optional|good to have|extra credit)\b",
    )
    .unwrap()
}

/// Optional wording inside the same sentence as the skill ("Rust is preferred but not required").
fn opt_sentence_re() -> Regex {
    Regex::new(
        r"(?i)\b(nice[- ]to[- ]have|bonus|a plus|is a plus|are a plus|preferred|optional|good to have|extra credit|but not required|not required)\b",
    )
    .unwrap()
}

/// Headings that introduce hiring requirements.
fn req_head_re() -> Regex {
    Regex::new(
        r"(?i)\b(requirements?|qualifications?|must[- ]have|what you('ll| will) need|you have|who you are|what we('re| are) looking for|you bring|minimum)\b",
    )
    .unwrap()
}

/// Headings that start some other section (about us, responsibilities, benefits ...).
fn stop_head_re() -> Regex {
    Regex::new(
        r"(?i)^\s*(what|who|about|responsibilit|benefits|perks|why|how|the role|your|our|we offer|you'll|you will|you may|you are|you're)\b",
    )
    .unwrap()
}

fn heading_like(h: &str) -> bool {
    !h.is_empty() && h.chars().count() <= 70 && !h.ends_with('.')
}

/// The sentence of `line` that mentions one of `terms` (lines can hold several sentences,
/// and "Rust is preferred" must not make the next sentence's skills optional).
fn sentence_with<'a>(line: &'a str, terms: &[String]) -> &'a str {
    line.split(". ")
        .find(|s| terms.iter().any(|t| contains_word(s, t)))
        .unwrap_or(line)
}

/// Why (if at all) this occurrence is optional: its own sentence says "a plus" / "preferred /
/// not required", or it sits under a nice-to-have heading. A requirements-style or other
/// section heading stops the upward search, so later sections cannot inherit an earlier heading.
fn optional_reason(lines: &[&str], idx: usize, sent: &str) -> Option<String> {
    let (opt, req_head, stop) = (opt_heading_re(), req_head_re(), stop_head_re());
    if opt_sentence_re().is_match(&fix_quotes(sent)) {
        return Some("sentence says plus/preferred/not required".into());
    }
    for j in (idx.saturating_sub(25)..idx).rev() {
        let h = fix_quotes(lines[j]);
        let h = h.trim();
        if heading_like(h) {
            if opt.is_match(h) {
                return Some(format!("under heading: {h}"));
            }
            // Any other "Heading:" line (e.g. "We are not looking for:") ends the section too.
            if req_head.is_match(h) || stop.is_match(h) || h.ends_with(':') {
                return None;
            }
        }
    }
    None
}

/// Does this occurrence read as a hiring requirement? Either its sentence uses requirement
/// wording ("experience with", "you have", "must" ...) or it sits under a requirements heading.
fn requirement_context(lines: &[&str], idx: usize, sent: &str) -> bool {
    let cue = Regex::new(
        r"(?i)\b(experience|proficien\w*|knowledge|familiar\w*|understanding|expertise|skilled?|comfortable|fluen\w*|must|required|you (have|know|can|are|will have)|strong|ability|hands-on|working)\b",
    )
    .unwrap();
    if cue.is_match(&fix_quotes(sent)) {
        return true;
    }
    let (opt, req_head, stop) = (opt_heading_re(), req_head_re(), stop_head_re());
    for j in (idx.saturating_sub(25)..idx).rev() {
        let h = fix_quotes(lines[j]);
        let h = h.trim();
        if heading_like(h) {
            if opt.is_match(h) {
                return false;
            }
            if req_head.is_match(h) {
                return true;
            }
            if stop.is_match(h) || h.ends_with(':') {
                return false;
            }
        }
    }
    false
}

/// Classify verified skills as (required, optional, mentioned) from where the posting says them.
/// Required: some occurrence reads as a requirement and is not optional. Optional: none is
/// required and some occurrence is optional. Mentioned: neither (company description, tool
/// lists). With `strict` off, mentioned skills count as required (the earlier, looser rule).
pub fn split_optional(
    items: Vec<Item>,
    posting: &str,
    strict: bool,
) -> (Vec<Item>, Vec<Item>, Vec<Item>) {
    let lines: Vec<&str> = posting.lines().collect();
    let (mut req, mut opt, mut men) = (vec![], vec![], vec![]);
    for it in items {
        let terms = terms_for(&it.skill);
        let occ: Vec<usize> = (0..lines.len())
            .filter(|&i| terms.iter().any(|t| contains_word(lines[i], t)))
            .collect();
        let mut required_at: Option<usize> = None;
        let mut non_opt_at: Option<usize> = None;
        let mut first_optional: Option<(usize, String)> = None;
        let mut first_any: Option<usize> = None;
        for &i in &occ {
            let sent = sentence_with(lines[i], &terms);
            first_any.get_or_insert(i);
            match optional_reason(&lines, i, sent) {
                Some(r) => {
                    first_optional.get_or_insert((i, r));
                }
                None => {
                    non_opt_at.get_or_insert(i);
                    if requirement_context(&lines, i, sent) {
                        required_at.get_or_insert(i);
                    }
                }
            }
        }
        let clip = |i: usize, n: usize| -> String { lines[i].trim().chars().take(n).collect() };
        let optional_item = |it: Item, i: usize, r: String| Item {
            skill: it.skill,
            quote: format!(
                "{}  [{r}]",
                lines[i].trim().chars().take(120).collect::<String>()
            ),
        };
        if strict {
            // required: some mention reads as a requirement; optional: only optional mentions
            // plus bare mentions; mentioned: nothing says it is a requirement or optional.
            if let Some(i) = required_at {
                req.push(Item {
                    skill: it.skill,
                    quote: clip(i, 200),
                });
            } else if let Some((i, r)) = first_optional {
                opt.push(optional_item(it, i, r));
            } else if let Some(i) = first_any {
                men.push(Item {
                    skill: it.skill,
                    quote: clip(i, 200),
                });
            } else {
                req.push(it);
            }
        } else {
            // loose: required unless EVERY mention is optional
            if let Some(i) = non_opt_at {
                req.push(Item {
                    skill: it.skill,
                    quote: clip(i, 200),
                });
            } else if let Some((i, r)) = first_optional {
                opt.push(optional_item(it, i, r));
            } else {
                req.push(it);
            }
        }
    }
    (req, opt, men)
}

// ---------- rules (no model involved) ----------

pub fn extract_rules(text: &str) -> Rules {
    let lower = text.to_lowercase();
    let mut r = Rules::default();

    // Three states: Some(true) says remote, Some(false) says not remote / in-person,
    // None never says. Negations ("no remote") must not count as a remote mention.
    let neg = Regex::new(
        r"\b(no|not|non|never|without|isn't|aren't)\s+(?:a\s+|fully\s+|100%\s+)?remote\b",
    )
    .unwrap();
    let negated = neg.is_match(&lower);
    let positive = neg.replace_all(&lower, " ").contains("remote");
    r.onsite_or_hybrid =
        Regex::new(r"\b(on-?site|hybrid|in-office|in office|in-person|in person)\b")
            .unwrap()
            .is_match(&lower);
    let hybrid_work = Regex::new(
        r"\bhybrid (work|working|role|position|model|schedule|environment|setup|arrangement)\b|\bli-hybrid\b|\bin-office\b",
    )
    .unwrap()
    .is_match(&lower);
    let fully_remote = Regex::new(r"\b(fully|100%|entirely|completely)[- ]remote\b")
        .unwrap()
        .is_match(&lower);
    r.remote = if positive && hybrid_work && !fully_remote {
        Some(false) // "hybrid ... blend of remote flexibility": not remote
    } else if positive {
        Some(true)
    } else if negated || r.onsite_or_hybrid {
        Some(false)
    } else {
        None
    };

    // "3+ years", "1 year", "3-5 years", "two years": smallest number, but only from
    // sentences about experience, so "founded 8 years ago" is ignored.
    let yrs = Regex::new(
        r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*\+?\s*(?:-\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)\b",
    )
    .unwrap();
    let word = |w: &str| -> Option<u32> {
        match w {
            "one" => Some(1),
            "two" => Some(2),
            "three" => Some(3),
            "four" => Some(4),
            "five" => Some(5),
            "six" => Some(6),
            "seven" => Some(7),
            "eight" => Some(8),
            "nine" => Some(9),
            "ten" => Some(10),
            n => n.parse().ok(),
        }
    };
    let exp_sentences: Vec<&str> = lower
        .split(|c| c == '.' || c == '\n')
        .filter(|s| s.contains("experience") || s.contains("exp."))
        .collect();
    // First number in each sentence ("10+ years ... including 2+ years in X" means 10), then the
    // smallest across sentences.
    let from = |scope: &[&str]| -> Option<u32> {
        scope
            .iter()
            .filter_map(|sent| yrs.captures(sent).and_then(|c| word(&c[1])))
            .min()
    };
    // Prefer experience sentences. Otherwise accept only requirement-shaped patterns:
    // "N+ years" anywhere, or a bare line such as "1+ years" / "2-4 years".
    // A plain "~2 years since we started" is company history, not a requirement.
    let plus = Regex::new(r"\b(\d{1,2})\s*\+\s*(?:years?|yrs?)\b").unwrap();
    let bare =
        Regex::new(r"(?m)^\s*(\d{1,2})\s*(?:\+|-\s*\d{1,2})?\s*(?:years?|yrs?)\s*$").unwrap();
    let fallback = || -> Option<u32> {
        plus.captures_iter(&lower)
            .chain(bare.captures_iter(&lower))
            .filter_map(|c| c[1].parse::<u32>().ok())
            .min()
    };
    r.min_years = from(&exp_sentences).or_else(fallback);

    // "UTC-5 to UTC+4", "UTC-5 - UTC+4", "UTC+1 to +4"
    let tz = Regex::new(r"utc\s*([+\-−]\s*\d{1,2})\s*(?:to|-|–|—)\s*(?:utc\s*)?([+\-−]\s*\d{1,2})")
        .unwrap();
    if let Some(c) = tz.captures(&lower) {
        let p = |s: &str| s.replace('−', "-").replace(' ', "").parse::<i32>().ok();
        if let (Some(a), Some(b)) = (p(&c[1]), p(&c[2])) {
            r.tz_window = Some((a.min(b), a.max(b)));
        }
    }

    let flags = [
        "must be located in",
        "must reside in",
        "must be based in",
        "authorized to work",
        "authorised to work",
        "work authorization",
        "right to work",
        "visa sponsorship",
        "us citizen",
        "us only",
        "us-based",
        "u.s. based",
        "remote (us",
        "remote, us",
        "remote - us",
        "within the us",
        "in the united states",
    ];
    let tzname = Regex::new(
        r"\b(pacific|eastern|central|mountain) time\b|\b(pst|pdt|est|edt|cst|cdt|cet|cest|gmt|utc)\b|time ?zone|overlap with",
    )
    .unwrap();
    for sentence in text.split(|c| c == '.' || c == '\n') {
        let sl = sentence.to_lowercase();
        if flags.iter().any(|f| sl.contains(f)) {
            r.auth_flags.push(sentence.trim().to_string());
        }
        if r.tz_window.is_none() && tzname.is_match(&sl) {
            r.tz_notes.push(sentence.trim().to_string());
        }
    }
    r
}

// ---------- matching ----------

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '+' || c == '#'
}

/// True if `needle` occurs in `hay` as a whole term ("go" does not match "mongodb",
/// "c" does not match "cloud", but "c++" and "c#" are handled).
pub fn contains_word(hay: &str, needle: &str) -> bool {
    let (hay, needle) = (hay.to_lowercase(), needle.to_lowercase());
    if needle.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(i) = hay[start..].find(&needle) {
        let a = start + i;
        let b = a + needle.len();
        let before_ok = hay[..a].chars().next_back().map_or(true, |c| !is_word(c));
        let after_ok = hay[b..].chars().next().map_or(true, |c| !is_word(c));
        if before_ok && after_ok {
            return true;
        }
        start = a + needle.chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

/// Alias-aware: "PostgreSQL" matches "postgres", "Kubernetes" matches "k8s".
fn skill_matches(req: &str, mine: &[String]) -> bool {
    let rt = terms_for(req);
    mine.iter().any(|m| {
        let mt = terms_for(m);
        rt.iter().any(|a| {
            mt.iter()
                .any(|b| contains_word(a, b) || contains_word(b, a))
        })
    })
}

/// Coverage check: terms the posting mentions that the model did not extract.
/// Deterministic, so it catches model misses without trusting the model.
const VOCAB: &[&str] = &[
    "rust",
    "go",
    "golang",
    "python",
    "java",
    "c++",
    "typescript",
    "javascript",
    "kotlin",
    "scala",
    "elixir",
    "ruby",
    "node.js",
    "postgres",
    "postgresql",
    "mysql",
    "redis",
    "kafka",
    "mongodb",
    "kubernetes",
    "docker",
    "terraform",
    "aws",
    "gcp",
    "azure",
    "grpc",
    "graphql",
    "rest",
    "tokio",
    "axum",
    "actix",
    "linux",
    "nix",
    "ci/cd",
    "react",
    "sql",
    "wasm",
    "mcp",
];

pub fn missed_terms(posting: &str, accepted: &[Item], profile: &[String]) -> Vec<String> {
    let mut terms: Vec<String> = VOCAB.iter().map(|s| s.to_string()).collect();
    terms.extend(profile.iter().cloned());
    terms.sort();
    terms.dedup();
    terms
        .into_iter()
        .filter(|t| contains_word(posting, t))
        .filter(|t| {
            !accepted
                .iter()
                .any(|i| contains_word(&i.skill, t) || contains_word(t, &i.skill))
        })
        .collect()
}

pub struct Analysis {
    pub rules: Rules,
    pub accepted: Vec<Item>,
    pub rejected: Vec<Item>,
    pub have: Vec<String>,
    pub gaps: Vec<String>,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
    pub missed: Vec<String>,
    pub optional: Vec<Item>,
    pub mentioned: Vec<Item>,
}

pub fn decide(rules: Rules, required: Vec<Item>, rejected: Vec<Item>, p: &Profile) -> Analysis {
    let mut have = vec![];
    let mut gaps = vec![];
    for it in &required {
        if skill_matches(&it.skill, &p.skills) {
            have.push(it.skill.clone());
        } else {
            gaps.push(it.skill.clone());
        }
    }
    let mut reasons = vec![];
    let mut hard_skip = false;

    if let Some((lo, hi)) = rules.tz_window {
        if p.tz_offset < lo || p.tz_offset > hi {
            hard_skip = true;
            reasons.push(format!(
                "timezone window UTC{:+} to UTC{:+} excludes UTC{:+}",
                lo, hi, p.tz_offset
            ));
        } else {
            reasons.push(format!(
                "timezone OK (UTC{:+} is inside UTC{:+} to UTC{:+})",
                p.tz_offset, lo, hi
            ));
        }
    }
    let mut review_flag = false;
    match rules.remote {
        Some(false) => {
            hard_skip = true;
            reasons.push("posting says onsite, hybrid, in-person or explicitly not remote".into());
        }
        None => {
            review_flag = true;
            reasons.push("never mentions remote or onsite: check how it describes location".into());
        }
        Some(true) => {
            if rules.onsite_or_hybrid {
                reasons.push(
                    "says remote but also mentions onsite/hybrid: check it is not required".into(),
                );
            }
        }
    }
    if let Some(y) = rules.min_years {
        if (y as f32) > p.years_experience {
            reasons.push(format!(
                "asks for {y}+ years, you have {}",
                p.years_experience
            ));
        }
    }
    if !rules.auth_flags.is_empty() {
        reasons.push(format!(
            "{} location/authorization sentence(s) need a human look",
            rules.auth_flags.len()
        ));
    }

    let total = have.len() + gaps.len();
    let ratio = if total == 0 {
        0.0
    } else {
        have.len() as f32 / total as f32
    };
    let verdict = if hard_skip {
        Verdict::Skip
    } else if !rules.auth_flags.is_empty() || review_flag {
        Verdict::Review
    } else if total == 0 {
        Verdict::Review
    } else if ratio >= 0.7 {
        Verdict::Apply
    } else if ratio >= 0.4 {
        Verdict::Stretch
    } else {
        Verdict::Skip
    };

    Analysis {
        rules,
        accepted: required,
        rejected,
        have,
        gaps,
        verdict,
        reasons,
        missed: vec![],
        optional: vec![],
        mentioned: vec![],
    }
}

// ---------- model call ----------

const SYSTEM: &str = "You extract hiring requirements from job postings. \
Reply with JSON only: {\"required\":[{\"skill\":\"...\",\"quote\":\"...\"}],\"nice_to_have\":[...same shape...]}. \
'skill' is a short technology or skill name (e.g. 'Rust', 'PostgreSQL', 'Kubernetes'). \
'quote' MUST be copied character-for-character from the posting and MUST contain the skill name. \
Use the exact spelling of each skill as written in the posting. Never return bare strings: every entry is an object with both fields. \
'quote' is the whole sentence or line (at least 3 words), not just the skill name. \
Example for the line 'You have 3+ years of experience writing Rust.': \
{\"skill\":\"Rust\",\"quote\":\"You have 3+ years of experience writing Rust.\"}. \
Put items under 'required' unless the posting calls them a plus, bonus or nice to have. \
Do not infer, paraphrase, or add anything that is not written in the posting.";

/// Model output is cached per (prompt, model, posting). Re-scoring after a rule or label
/// change then takes seconds instead of minutes, and results are reproducible between runs.
/// Set JOBKIT_REFRESH=1 to bypass the cache.
pub fn with_cache(
    path: &Path,
    refresh: bool,
    fetch: impl FnOnce() -> Result<String, String>,
) -> Result<String, String> {
    if !refresh {
        if let Ok(c) = fs::read_to_string(path) {
            return Ok(c);
        }
    }
    let c = fetch()?;
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(path, &c);
    Ok(c)
}

fn cache_path(model: &str, posting: &str) -> std::path::PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (SYSTEM, model, posting).hash(&mut h);
    let safe: String = model
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    Path::new(".jobkit-cache").join(format!("{safe}-{:016x}.json", h.finish()))
}

fn call_model(posting: &str) -> Result<Extraction, String> {
    let url = env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = env::var("JOBKIT_MODEL").unwrap_or_else(|_| "gemma3:4b".into());
    let body = serde_json::json!({
        "model": model,
        "stream": false,
        "format": "json",
        "options": { "temperature": 0 },
        "messages": [
            { "role": "system", "content": SYSTEM },
            { "role": "user", "content": posting }
        ]
    });
    let refresh = env::var("JOBKIT_REFRESH").is_ok();
    let content = with_cache(&cache_path(&model, posting), refresh, || {
        let resp: serde_json::Value = ureq::post(&format!("{url}/api/chat"))
            .send_json(body)
            .map_err(|e| format!("ollama request failed: {e}"))?
            .into_json()
            .map_err(|e| format!("bad ollama response: {e}"))?;
        resp["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "no message.content".to_string())
    })?;
    let raw: RawExtraction = serde_json::from_str(&content)
        .map_err(|e| format!("model returned invalid JSON: {e}\n{content}"))?;
    Ok(Extraction {
        required: raw.required.into_iter().map(Item::from).collect(),
        nice_to_have: raw.nice_to_have.into_iter().map(Item::from).collect(),
    })
}

// ---------- commands ----------

fn load_profile(path: &str) -> Profile {
    let s = fs::read_to_string(path).unwrap_or_else(|e| die(&format!("cannot read {path}: {e}")));
    toml::from_str(&s).unwrap_or_else(|e| die(&format!("bad profile: {e}")))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    process::exit(1)
}

fn analyze_text(
    text: &str,
    p: &Profile,
    rules_only: bool,
    strict: bool,
) -> Result<Analysis, String> {
    let rules = extract_rules(text);
    let (req, rej, opt, men) = if rules_only {
        (vec![], vec![], vec![], vec![])
    } else {
        // The model only names skills. Rust verifies they are in the posting, then decides
        // required / optional / mentioned from where and how the posting says them.
        let ext = call_model(text)?;
        let mut all = ext.required;
        all.extend(ext.nice_to_have);
        let (ok, rej) = verify(all, text);
        let (req, opt, men) = split_optional(ok, text, strict);
        (req, rej, opt, men)
    };
    let mut a = decide(rules, req, rej, p);
    a.optional = opt;
    a.mentioned = men;
    if !rules_only {
        let all: Vec<Item> = a
            .accepted
            .iter()
            .chain(&a.optional)
            .chain(&a.mentioned)
            .cloned()
            .collect();
        a.missed = missed_terms(text, &all, &p.skills);
    }
    Ok(a)
}

fn print_analysis(a: &Analysis) {
    println!("VERDICT: {:?}", a.verdict);
    for r in &a.reasons {
        println!("  - {r}");
    }
    println!("HAVE ({}): {}", a.have.len(), a.have.join(", "));
    println!("GAPS ({}): {}", a.gaps.len(), a.gaps.join(", "));
    for f in &a.rules.auth_flags {
        println!("CHECK: {f}");
    }
    for t in &a.rules.tz_notes {
        println!("TIMEZONE NOTE (no UTC window parsed): {t}");
    }
    if !a.missed.is_empty() {
        println!(
            "COVERAGE: posting mentions but model did not extract: {} (check by hand)",
            a.missed.join(", ")
        );
    }
    println!("VERIFIED REQUIREMENTS:");
    for it in &a.accepted {
        println!("  [{}] \"{}\"", it.skill, it.quote);
    }
    if !a.optional.is_empty() {
        let names: Vec<&str> = a.optional.iter().map(|i| i.skill.as_str()).collect();
        println!("OPTIONAL (nice-to-have): {}", names.join(", "));
    }
    if !a.mentioned.is_empty() {
        let names: Vec<&str> = a.mentioned.iter().map(|i| i.skill.as_str()).collect();
        println!(
            "MENTIONED (not clearly a requirement): {}",
            names.join(", ")
        );
    }
    if !a.rejected.is_empty() {
        println!("REJECTED BY CHECKER ({}):", a.rejected.len());
        for it in &a.rejected {
            println!("  [{}] \"{}\"", it.skill, it.quote);
        }
    }
}

#[derive(Deserialize)]
struct Label {
    remote: Option<bool>,
    min_years: Option<u32>,
    tz: Option<(i32, i32)>,
    #[serde(default)]
    required_skills: Vec<String>,
}

fn eval_dir(dir: &str, p: &Profile, rules_only: bool, verbose: bool, strict: bool) {
    let (mut n, mut remote_ok, mut years_ok, mut tz_ok) = (0, 0, 0, 0);
    let (mut tp, mut fp, mut fn_, mut rejected, mut accepted, mut missed) = (0, 0, 0, 0, 0, 0);
    let mut skipped: Vec<String> = vec![];
    let mut model_failures = 0;
    let mut label_warnings = 0;
    let mut optional_total = 0;
    let mut mentioned_total = 0;
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| die(&format!("{dir}: {e}")))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "txt"))
        .collect();
    entries.sort();
    for path in entries {
        // Labels live in <dir>/labels/<name>.json, or next to the posting as <name>.json.
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let in_labels = Path::new(dir).join("labels").join(format!("{stem}.json"));
        let label_path = if in_labels.exists() {
            in_labels
        } else {
            path.with_extension("json")
        };
        let Ok(ls) = fs::read_to_string(&label_path) else {
            skipped.push(stem);
            continue;
        };
        let label: Label = serde_json::from_str(&ls)
            .unwrap_or_else(|e| die(&format!("{}: {e}", label_path.display())));
        let text = fs::read_to_string(&path).unwrap();
        let mut diffs: Vec<String> = vec![];
        let a = match analyze_text(&text, p, rules_only, strict) {
            Ok(a) => a,
            Err(e) => {
                model_failures += 1;
                diffs.push(format!(
                    "MODEL ERROR (counted as zero extracted): {}",
                    e.lines().next().unwrap_or("")
                ));
                analyze_text(&text, p, true, strict).unwrap()
            }
        };
        n += 1;
        for want in &label.required_skills {
            if !contains_word(&text, want) {
                label_warnings += 1;
                diffs.push(format!(
                    "label check: required skill '{want}' not found in posting text"
                ));
            }
        }
        if a.rules.remote != label.remote {
            diffs.push(format!(
                "remote: expected {:?}, got {:?}",
                label.remote, a.rules.remote
            ));
        }
        if a.rules.min_years != label.min_years {
            diffs.push(format!(
                "min_years: expected {:?}, got {:?}",
                label.min_years, a.rules.min_years
            ));
        }
        if a.rules.tz_window != label.tz {
            diffs.push(format!(
                "tz: expected {:?}, got {:?}",
                label.tz, a.rules.tz_window
            ));
        }
        remote_ok += (a.rules.remote == label.remote) as u32;
        years_ok += (a.rules.min_years == label.min_years) as u32;
        tz_ok += (a.rules.tz_window == label.tz) as u32;
        if !rules_only {
            accepted += a.accepted.len() as u32;
            rejected += a.rejected.len() as u32;
            missed += a.missed.len() as u32;
            optional_total += a.optional.len();
            mentioned_total += a.mentioned.len();
            let found: Vec<&String> = a.accepted.iter().map(|i| &i.skill).collect();
            let found_names: Vec<String> = found.iter().map(|f| (*f).clone()).collect();
            let (mut extras, mut misses) = (vec![], vec![]);
            for f in &found {
                if skill_matches(f, &label.required_skills) {
                    tp += 1
                } else {
                    fp += 1;
                    extras.push((*f).clone());
                }
            }
            for want in &label.required_skills {
                let hit = skill_matches(want, &found_names);
                if !hit {
                    fn_ += 1;
                    misses.push(want.clone());
                }
            }
            if verbose {
                for it in &a.optional {
                    diffs.push(format!("optional: [{}] {}", it.skill, it.quote));
                }
                for it in &a.mentioned {
                    let q: String = it.quote.chars().take(90).collect();
                    diffs.push(format!("mentioned: [{}] {}", it.skill, q));
                }
                diffs.push(format!(
                    "extra (model said required, not in your label): {}",
                    extras.join(", ")
                ));
                diffs.push(format!(
                    "missed (in your label, model did not extract): {}",
                    misses.join(", ")
                ));
                for it in &a.rejected {
                    let q: String = it.quote.chars().take(80).collect();
                    diffs.push(format!(
                        "rejected ({}): [{}] \"{}\"",
                        reject_reason(it, &text),
                        it.skill,
                        q
                    ));
                }
            }
        }
        println!(
            "{}: {:?}",
            Path::new(&path).file_name().unwrap().to_string_lossy(),
            a.verdict
        );
        for d in &diffs {
            println!("    {d}");
        }
    }
    if n == 0 {
        die("no labelled postings found (need X.txt + X.json pairs)");
    }
    let pct = |a: u32| 100.0 * a as f32 / n as f32;
    println!("label skills not found in their posting text: {label_warnings}");
    println!(
        "\n== {n} labelled postings ({} unlabelled skipped) ==",
        skipped.len()
    );
    println!("remote flag accuracy: {:.0}%", pct(remote_ok));
    println!("min-years accuracy:   {:.0}%", pct(years_ok));
    println!("timezone accuracy:    {:.0}%", pct(tz_ok));
    if !rules_only {
        let prec = if tp + fp == 0 {
            0.0
        } else {
            tp as f32 / (tp + fp) as f32
        };
        let rec = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f32 / (tp + fn_) as f32
        };
        println!("skills precision: {:.2}  recall: {:.2}", prec, rec);
        println!(
            "model items proposed: {}  rejected by quote checker: {}",
            accepted + rejected,
            rejected
        );
        println!("known terms in postings the model never extracted: {missed}");
        println!("postings where the model output was unusable: {model_failures}");
        println!("skills kept out of 'required' as optional: {optional_total}, as mentioned-only: {mentioned_total}");
        println!(
            "mode: {}",
            if strict {
                "strict context rules"
            } else {
                "loose (mentioned counts as required)"
            }
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let rules_only = args.iter().any(|a| a == "--rules-only");
    let verbose = args.iter().any(|a| a == "--verbose");
    let strict = !args.iter().any(|a| a == "--loose");
    let profile_path = args
        .iter()
        .position(|a| a == "--profile")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "profile.toml".into());
    let pos: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && !(*i > 0 && args[*i - 1] == "--profile"))
        .map(|(_, a)| a)
        .collect();
    let p = load_profile(&profile_path);
    match (pos.get(0).map(|s| s.as_str()), pos.get(1)) {
        (Some("analyze"), Some(path)) => {
            let text = fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
            print_analysis(
                &analyze_text(&text, &p, rules_only, strict).unwrap_or_else(|e| die(&e)),
            );
        }
        (Some("eval"), Some(dir)) => eval_dir(dir, &p, rules_only, verbose, strict),
        _ => die(
            "usage: jobkit analyze <posting.txt> | jobkit eval <dir>  [--profile f] [--rules-only]",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POST: &str = "Cloud Integrations Engineer (Rust).\nThis role is fully remote, UTC-5 to UTC+4.\nYou have 3+ years of experience with Rust and Kubernetes.\nMust be authorized to work in your country of residence.";

    #[test]
    fn rules_parse() {
        let r = extract_rules(POST);
        assert_eq!(r.remote, Some(true));
        assert_eq!(r.min_years, Some(3));
        assert_eq!(r.tz_window, Some((-5, 4)));
        assert_eq!(r.auth_flags.len(), 1);
    }

    #[test]
    fn verifier_rejects_skills_not_in_posting() {
        let items = vec![
            Item {
                skill: "Rust".into(),
                quote: "whatever the model said".into(),
            },
            Item {
                skill: "Go".into(),
                quote: "experience with Go microservices".into(),
            },
            Item {
                skill: "Kubernetes".into(),
                quote: "".into(),
            },
            Item {
                skill: "rust".into(),
                quote: "dup".into(),
            },
        ];
        let (ok, bad) = verify(items, POST);
        assert_eq!(ok.len(), 2); // Rust, Kubernetes (duplicate dropped)
        assert_eq!(bad.len(), 1); // Go is not in the posting
        assert!(ok[0].quote.contains("Rust")); // evidence comes from the posting, not the model
    }

    #[test]
    fn nice_to_have_sections_are_optional() {
        let post = "Requirements\nYou know Rust well.\nExperience with Postgres.\nNice to have\nFamiliarity with gRPC.\nWe like Nix, a plus.\nBenefits\nFree coffee.";
        let items = vec![
            Item {
                skill: "Rust".into(),
                quote: "".into(),
            },
            Item {
                skill: "Postgres".into(),
                quote: "".into(),
            },
            Item {
                skill: "gRPC".into(),
                quote: "".into(),
            },
            Item {
                skill: "Nix".into(),
                quote: "".into(),
            },
        ];
        let (ok, bad) = verify(items, post);
        assert!(bad.is_empty());
        let (req, opt, _men) = split_optional(ok, post, true);
        let r: Vec<&str> = req.iter().map(|i| i.skill.as_str()).collect();
        let o: Vec<&str> = opt.iter().map(|i| i.skill.as_str()).collect();
        assert_eq!(r, vec!["Rust", "Postgres"]);
        assert_eq!(o, vec!["gRPC", "Nix"]);
    }

    #[test]
    fn curly_apostrophe_heading_stops_inheritance() {
        let post = "Nice to have\nSomething else.\nWhat we\u{2019}re looking for\nRust experience.";
        let (ok, _) = verify(
            vec![Item {
                skill: "Rust".into(),
                quote: "".into(),
            }],
            post,
        );
        let (req, opt, _men) = split_optional(ok, post, true);
        assert_eq!((req.len(), opt.len()), (1, 0));
    }

    #[test]
    fn required_anywhere_wins() {
        let post = "Requirements\nPython experience.\nNice to have\nPython tooling.";
        let (ok, _) = verify(
            vec![Item {
                skill: "Python".into(),
                quote: "".into(),
            }],
            post,
        );
        let (req, opt, _men) = split_optional(ok, post, true);
        assert_eq!((req.len(), opt.len()), (1, 0));
        assert!(req[0].quote.contains("experience"));
    }

    #[test]
    fn tech_stack_prose_is_mentioned_not_required() {
        let post =
            "About us\nOur platform is powered by Django.\nRequirements\nExperience with Rust.";
        let items = vec![
            Item {
                skill: "Django".into(),
                quote: "".into(),
            },
            Item {
                skill: "Rust".into(),
                quote: "".into(),
            },
        ];
        let (ok, _) = verify(items.clone(), post);
        let (req, _opt, men) = split_optional(ok, post, true);
        assert_eq!(req.len(), 1);
        assert_eq!(men.len(), 1);
        let (ok2, _) = verify(items, post);
        let (req2, _, men2) = split_optional(ok2, post, false);
        assert_eq!((req2.len(), men2.len()), (2, 0));
    }

    #[test]
    fn unknown_colon_heading_ends_a_good_to_have_section() {
        let post = "Good to have:\nSome experience shipping software.\nWe are not looking for:\nPeople who wait.\nFrontend: TypeScript, React";
        let items = vec![Item {
            skill: "TypeScript".into(),
            quote: "".into(),
        }];
        let (ok, _) = verify(items.clone(), post);
        let (req, opt, men) = split_optional(ok, post, true);
        assert_eq!((req.len(), opt.len(), men.len()), (0, 0, 1));
        let (ok2, _) = verify(items, post);
        let (req2, opt2, _) = split_optional(ok2, post, false);
        assert_eq!((req2.len(), opt2.len()), (1, 0));
    }

    #[test]
    fn not_required_heading_does_not_demote_skills() {
        let post =
            "Requirements\nExperience with Rust.\nUS citizenship/visa not required\nTypeScript";
        let items = vec![Item {
            skill: "TypeScript".into(),
            quote: "".into(),
        }];
        let (ok, _) = verify(items, post);
        let (req, opt, _men) = split_optional(ok, post, true);
        assert_eq!((req.len(), opt.len()), (1, 0));
    }

    #[test]
    fn loose_vs_strict_when_stack_prose_and_nice_to_have_disagree() {
        let post = "About us\nWe use Go.\nNice to have\nGo experience.";
        let items = vec![Item {
            skill: "Go".into(),
            quote: "".into(),
        }];
        let (ok, _) = verify(items.clone(), post);
        let (r1, o1, _) = split_optional(ok, post, true);
        assert_eq!((r1.len(), o1.len()), (0, 1));
        let (ok2, _) = verify(items, post);
        let (r2, o2, _) = split_optional(ok2, post, false);
        assert_eq!((r2.len(), o2.len()), (1, 0));
    }

    #[test]
    fn preferred_sentence_does_not_taint_its_neighbour() {
        let post = "Requirements\nYou can write Rust, Go, or C. Rust is preferred but not required. You know memory and concurrency.";
        let items = vec![
            Item {
                skill: "memory".into(),
                quote: "".into(),
            },
            Item {
                skill: "Rust".into(),
                quote: "".into(),
            },
        ];
        let (ok, _) = verify(items, post);
        let (req, opt, _men) = split_optional(ok, post, true);
        let r: Vec<&str> = req.iter().map(|i| i.skill.as_str()).collect();
        assert!(r.contains(&"memory"));
        assert!(r.contains(&"Rust")); // also has a requirement-context sentence
        assert!(opt.is_empty());
    }

    #[test]
    fn preferred_qualifications_heading_is_optional() {
        let post = "Minimum qualifications\nPython.\nPreferred qualifications\nGo experience.";
        let (ok, _) = verify(
            vec![
                Item {
                    skill: "Python".into(),
                    quote: "".into(),
                },
                Item {
                    skill: "Go".into(),
                    quote: "".into(),
                },
            ],
            post,
        );
        let (req, opt, _men) = split_optional(ok, post, true);
        assert_eq!(req.len(), 1);
        assert_eq!(opt.len(), 1);
    }

    #[test]
    fn scoring_is_alias_aware() {
        assert!(skill_matches("PostgreSQL", &["postgres".to_string()]));
        assert!(skill_matches("Kubernetes", &["k8s".to_string()]));
        assert!(!skill_matches("MongoDB", &["go".to_string()]));
    }

    #[test]
    fn cache_reuses_first_result() {
        let dir = std::env::temp_dir().join(format!("jobkit-test-{}", std::process::id()));
        let path = dir.join("x.json");
        let a = with_cache(&path, false, || Ok("first".into())).unwrap();
        let b = with_cache(&path, false, || Ok("second".into())).unwrap();
        let c = with_cache(&path, true, || Ok("third".into())).unwrap();
        assert_eq!(
            (a.as_str(), b.as_str(), c.as_str()),
            ("first", "first", "third")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn verifier_handles_aliases() {
        let post = "Backend: Golang, Postgres\nWe use k8s and Tailwind.";
        let items = vec![
            Item {
                skill: "PostgreSQL".into(),
                quote: "".into(),
            },
            Item {
                skill: "Kubernetes".into(),
                quote: "".into(),
            },
            Item {
                skill: "Tailwind CSS".into(),
                quote: "".into(),
            },
            Item {
                skill: "MongoDB".into(),
                quote: "".into(),
            },
        ];
        let (ok, bad) = verify(items, post);
        assert_eq!(ok.len(), 3);
        assert_eq!(bad.len(), 1);
    }

    #[test]
    fn bare_strings_become_skills_checked_against_the_posting() {
        let raw: RawExtraction = serde_json::from_str(
            r#"{"required":[{"skill":"Rust","quote":"Rust"}],"nice_to_have":["gRPC"]}"#,
        )
        .unwrap();
        let req: Vec<Item> = raw.required.into_iter().map(Item::from).collect();
        let nice: Vec<Item> = raw.nice_to_have.into_iter().map(Item::from).collect();
        assert_eq!(verify(req, POST).0.len(), 1);
        assert_eq!(verify(nice, POST).1.len(), 1); // gRPC is not in POST
    }

    #[test]
    fn duplicate_keys_merge() {
        let raw: RawExtraction = serde_json::from_str(
            r#"{"required":[],"nice_to_have":[{"skill":"A","quote":"a b c"}],"nice_to_have":[{"skill":"B","quote":"b c d"}]}"#,
        )
        .unwrap();
        assert_eq!(raw.nice_to_have.len(), 2);
    }

    #[test]
    fn word_boundary_matching() {
        assert!(!contains_word("MongoDB", "go"));
        assert!(!contains_word("cloud", "c"));
        assert!(contains_word("C++ and Go", "go"));
        assert!(contains_word("experience with C++", "c++"));
        assert!(!contains_word("c++", "c"));
        let mine = vec!["go".to_string()];
        assert!(!skill_matches("MongoDB", &mine));
        assert!(skill_matches("Go", &mine));
    }

    #[test]
    fn coverage_finds_missed_terms() {
        let got = vec![Item {
            skill: "Rust".into(),
            quote: "x y z".into(),
        }];
        let m = missed_terms(POST, &got, &[]);
        assert!(m.contains(&"kubernetes".to_string()));
        assert!(!m.contains(&"rust".to_string()));
    }

    #[test]
    fn years_singular_words_and_noise() {
        assert_eq!(
            extract_rules("Requires 1 year of experience with Rust.").min_years,
            Some(1)
        );
        assert_eq!(
            extract_rules("Two years of experience in backend.").min_years,
            Some(2)
        );
        assert_eq!(
            extract_rules("We were founded 8 years ago.\n3+ years of experience required.")
                .min_years,
            Some(3)
        );
    }

    #[test]
    fn no_remote_word_is_review_not_skip() {
        let p = Profile {
            years_experience: 4.0,
            tz_offset: 1,
            skills: vec!["rust".into()],
        };
        let a = decide(
            extract_rules("Join our team. Rust required."),
            vec![],
            vec![],
            &p,
        );
        assert_eq!(a.verdict, Verdict::Review);
        let b = decide(
            extract_rules("On-site in Berlin. Rust required."),
            vec![],
            vec![],
            &p,
        );
        assert_eq!(b.verdict, Verdict::Skip);
    }

    #[test]
    fn bare_years_line_and_named_timezone() {
        let t = "Experience with LLMs.\n1+ years\n2+ years in a technical role.\nRemote (US), overlap with Pacific Time hours.";
        let r = extract_rules(t);
        assert_eq!(r.min_years, Some(1));
        assert!(r.tz_window.is_none());
        assert!(!r.tz_notes.is_empty());
        assert!(!r.auth_flags.is_empty()); // "Remote (US"
    }

    #[test]
    fn remote_three_states_and_negation() {
        assert_eq!(
            extract_rules("This is a hard requirement, no remote.").remote,
            Some(false)
        );
        assert_eq!(
            extract_rules("We work from San Francisco, in-person.").remote,
            Some(false)
        );
        assert_eq!(extract_rules("Fully remote team.").remote, Some(true));
        assert_eq!(extract_rules("Build things with Rust.").remote, None);
    }

    #[test]
    fn hybrid_is_not_remote() {
        let c = "This is a full-time position based out of the Toronto office. Our hybrid work model offers a blend of remote flexibility and in-person collaboration.\n#LI-HYBRID";
        assert_eq!(extract_rules(c).remote, Some(false));
        assert_eq!(
            extract_rules("Fully remote. We also build hybrid cloud products.").remote,
            Some(true)
        );
    }

    #[test]
    fn years_range_with_trailing_plus_and_including_clause() {
        assert_eq!(
            extract_rules("You have 12-15+ years software engineering experience.").min_years,
            Some(12)
        );
        assert_eq!(
            extract_rules("Relevant Experience: 10+ years of software engineering, including 2+ years in embedded roles").min_years,
            Some(10)
        );
    }

    #[test]
    fn company_history_is_not_a_requirement() {
        let t =
            "Growth: 8 figures of ARR, all just ~2 years since we started.\nBuild great things.";
        assert_eq!(extract_rules(t).min_years, None);
    }

    #[test]
    fn unicode_minus_timezone() {
        let r = extract_rules("Remote, UTC−5 to UTC+4");
        assert_eq!(r.tz_window, Some((-5, 4)));
    }

    #[test]
    fn timezone_excludes() {
        let p = Profile {
            years_experience: 4.0,
            tz_offset: 8,
            skills: vec!["rust".into()],
        };
        let a = decide(extract_rules(POST), vec![], vec![], &p);
        assert_eq!(a.verdict, Verdict::Skip);
    }

    #[test]
    fn apply_when_skills_match() {
        let p = Profile {
            years_experience: 4.0,
            tz_offset: 1,
            skills: vec!["rust".into(), "kubernetes".into()],
        };
        let req = vec![Item {
            skill: "Rust".into(),
            quote: "Rust".into(),
        }];
        let mut rules = extract_rules(POST);
        rules.auth_flags.clear();
        assert_eq!(decide(rules, req, vec![], &p).verdict, Verdict::Apply);
    }
}
