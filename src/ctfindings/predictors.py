"""Model requests for every non-encoder condition (paper section 2.3-2.4).

Each predictor turns one case into one result:
  {"probabilities": [18 floats or None] | None, "predictions": [18 ints or None],
   "status": "success" | "invalid", "requests", "input_tokens", "output_tokens", "model"}
Request errors raise; the runner retries once and then records "failed".
"""

import hashlib
import json
import math
import os

from .data import report_text

# ---------------------------------------------------------------- prompt text


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def display(label, language, ja_names):
    return f"{ja_names[label]}（{label}）" if language == "ja" else label


def question(label, language, prompt, ja_names):
    """One finding question: the shared rule plus the J1 wording."""
    return (prompt["rules"][language] + "\n"
            + prompt["question"][language].format(label=display(label, language, ja_names)))


def generation_messages(case, text, language, labels, ja_names, prompt, shots=()):
    system = (prompt["rules"][language] + "\n" + prompt["generated_output"][language] + "\n"
              + prompt["labels_heading"][language]
              + canonical([display(x, language, ja_names) for x in labels]))
    messages = [{"role": "system", "content": system}]
    shot_text = "en" if language == "en" else "ja_MT"
    for example in shots:
        messages += [
            {"role": "user", "content": report_text(example, shot_text, prompt["report_headings"])},
            {"role": "assistant", "content": canonical(dict(zip(labels, example["labels"],
                                                                strict=True)))}]
    messages.append({"role": "user", "content": report_text(case, text, prompt["report_headings"])})
    return messages


def output_schema(labels):
    return {"type": "object",
            "properties": {x: {"type": "integer", "enum": [0, 1]} for x in labels},
            "required": labels, "additionalProperties": False}


def parse_binary(content, labels):
    """Exactly the 18 keys with integer 0/1 values, else None."""
    def no_duplicates(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("duplicate key")
        return dict(pairs)
    try:
        value = json.loads(content, object_pairs_hook=no_duplicates)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != set(labels):
        return None
    if any(type(value[x]) is not int or value[x] not in (0, 1) for x in labels):
        return None
    return [value[x] for x in labels]


def valid_probability(value):
    ok = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    return float(value) if ok else None


def result(predictions, probabilities, usage, model, requests=1):
    complete = all(v is not None for v in predictions)
    return {"status": "success" if complete else "invalid", "predictions": predictions,
            "probabilities": probabilities, "requests": requests,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "model": model}


def from_probabilities(probabilities, usage, model, requests=1):
    predictions = [None if p is None else int(p >= 0.5) for p in probabilities]
    return result(predictions, probabilities, usage, model, requests)


def require_env(name):
    if not os.environ.get(name):
        raise PermissionError(f"Set {name} in the environment or .env")


# ---------------------------------------------------------------- predictors


class OpenAIGeneration:
    """gpt-6-luna, Responses API with strict JSON schema (G1-G4)."""
    env = "OPENAI_API_KEY"

    def __init__(self, condition, max_tokens=8192, client=None):
        self.condition, self.max_tokens, self.client = condition, max_tokens, client

    def request(self, messages, labels):
        effort = self.condition["reasoning_effort"]
        body = {"model": self.condition["model"], "input": messages,
                "reasoning": {"effort": effort},
                "text": {"format": {"type": "json_schema", "name": "ct_findings", "strict": True,
                                    "schema": output_schema(labels)}},
                "store": False, "tools": [], "truncation": "disabled",
                "max_output_tokens": self.max_tokens}
        if effort == "none":
            body["temperature"] = 0
        return body

    def __call__(self, body, labels):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(max_retries=0, timeout=120)
        response = self.client.responses.create(**body)
        raw = response.model_dump(mode="json")
        usage = raw.get("usage") or {}
        values = parse_binary(response.output_text, labels) if raw.get("status") == "completed" \
            else None
        return result(values or [None] * len(labels), None, usage, raw.get("model"))


class LocalGeneration:
    """Qwen and MedGemma on a vLLM OpenAI-compatible server (L3-L8, M1-M2)."""
    env = None

    def __init__(self, condition, endpoint, runtime, seed_salt=None, client=None):
        self.condition, self.endpoint, self.runtime = condition, endpoint.rstrip("/"), runtime
        self.seed_salt, self.client = seed_salt, client
        self.timeout = (runtime["max_output_tokens"] / runtime["min_output_tokens_per_second"]
                        + runtime["request_overhead_seconds"])

    def request(self, messages, labels, case_id=""):
        s = self.condition["sampling"]
        body = {"model": self.condition["model"], "messages": messages,
                "temperature": s.get("temperature", 0), "top_p": 1, "n": 1,
                "max_tokens": self.runtime["max_output_tokens"],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "ct_findings", "strict": True, "schema": output_schema(labels)}}}
        if "enable_thinking" in s:  # Qwen: all recommended sampling values are sent explicitly
            body["chat_template_kwargs"] = {"enable_thinking": s["enable_thinking"]}
            for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                        "repetition_penalty"):
                body[key] = s[key]
            if s["enable_thinking"]:
                body["reasoning_effort"] = s["reasoning_effort"]
                body["chat_template_kwargs"]["reasoning_effort"] = s["reasoning_effort"]
        if self.seed_salt:
            digest = hashlib.sha256(f"{self.seed_salt}|{case_id}".encode()).hexdigest()
            body["seed"] = int(digest[:8], 16) & 0x7FFFFFFF
        return body

    def __call__(self, body, labels):
        import httpx
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout, trust_env=False)
        response = self.client.post(self.endpoint + "/v1/chat/completions", json=body)
        response.raise_for_status()
        raw = response.json()
        usage = raw.get("usage") or {}
        usage = {"input_tokens": usage.get("prompt_tokens"),
                 "output_tokens": usage.get("completion_tokens")}
        values = None
        choices = raw.get("choices") or []
        if len(choices) == 1 and choices[0].get("finish_reason") == "stop":
            values = parse_binary(choices[0]["message"].get("content"), labels)
        return result(values or [None] * len(labels), None, usage, raw.get("model"))


class Jev:
    """Jev typed decisions: 18 questions in one system_one request (J1)."""
    env = "TYPESAFE_API_KEY"

    def __init__(self, condition, client=None):
        self.condition, self.client = condition, client

    def request(self, state, questions):
        return {"model": self.condition["model"], "state": state, "questions": questions}

    def __call__(self, body, labels):
        from pydantic import BaseModel, ConfigDict, Field
        from typesafe_sdk import Noul

        class Raw(BaseModel):  # keep every answer field as returned
            model_config = ConfigDict(extra="allow")
            model: str | None = None
            usage: dict = Field(default_factory=dict)
            answers: dict = Field(default_factory=dict)

        if self.client is None:
            from typesafe_sdk import TypeSafeClient
            from typesafe_sdk._core.retry import RetryPolicy
            self.client = TypeSafeClient(model=body["model"], retry=RetryPolicy(max_retries=0),
                                         timeout=120)
        questions = {x: Noul(instructions=q) for x, q in body["questions"].items()}
        raw = self.client.system_one(body["state"], questions, model=body["model"],
                                     response_model=Raw).model_dump(mode="json")
        answers = raw.get("answers") or {}
        probabilities = []
        for x in labels:
            answer = answers.get(x)
            ok = isinstance(answer, dict) and answer.get("type") == "noul"
            probabilities.append(valid_probability(answer.get("noul")) if ok else None)
        if raw.get("model") != body["model"]:  # pinned version must answer
            probabilities = [None] * len(labels)
        return from_probabilities(probabilities, raw.get("usage") or {}, raw.get("model"))


class Decisions:
    """OpenAI Decisions API: 18 named predicates in one request (D1)."""
    env = "OPENAI_API_KEY"

    def __init__(self, condition, client=None):
        self.condition, self.client = condition, client

    def request(self, state, questions):
        return {"model": self.condition["model"], "input": state,
                "questions": [{"type": "predicate", "name": x, "instructions": q}
                              for x, q in questions.items()]}

    def __call__(self, body, labels):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(max_retries=0, timeout=120)
        raw = self.client.decisions.create(**body).model_dump(mode="json")
        answers = raw.get("answers")
        probabilities = [None] * len(labels)
        # Answers keep the order and echo the names; anything else is left missing.
        if raw.get("model") and isinstance(answers, list) and len(answers) == len(labels) and all(
                isinstance(a, dict) and a.get("name") == x for a, x in zip(answers, labels)):
            probabilities = [valid_probability(a.get("probability"))
                             if a.get("type") == "predicate" else None for a in answers]
        return from_probabilities(probabilities, raw.get("usage") or {}, raw.get("model"))


class TypedQwen:
    """Typed decision on Qwen: first-token Yes/No probability, one request per finding (T1).

    Full-vocabulary log probabilities at the first output position; no token filtering.
    Single-token spellings of Yes/No are summed and normalised.
    """
    env = None
    SPELLINGS = {"yes": ["Yes", " Yes", "yes", " yes"], "no": ["No", " No", "no", " no"]}

    def __init__(self, condition, endpoint, runtime, tokenizer=None, client=None):
        self.condition, self.endpoint, self.runtime = condition, endpoint.rstrip("/"), runtime
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(condition["model"],
                                                      revision=condition["revision"])
        self.tokenizer, self.client = tokenizer, client
        self.candidates = {}
        for answer, spellings in self.SPELLINGS.items():
            ids = [tokenizer.encode(s, add_special_tokens=False) for s in spellings]
            self.candidates[answer] = sorted({i[0] for i in ids if len(i) == 1})
        if not all(self.candidates.values()) or set(self.candidates["yes"]) & set(
                self.candidates["no"]):
            raise ValueError("Yes/No token candidates are missing or overlap")

    def request(self, content):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)
        if len(ids) >= self.runtime["context_length"]:
            raise ValueError("Input exceeds the context length")
        return {"model": self.condition["model"], "prompt": ids, "max_tokens": 1,
                "temperature": 1, "top_p": 1, "top_k": -1, "min_p": 0, "repetition_penalty": 1,
                "presence_penalty": 0, "frequency_penalty": 0, "logprobs": 1,
                "logprob_token_ids": self.candidates["yes"] + self.candidates["no"],
                "return_tokens_as_token_ids": True, "add_special_tokens": False}

    def probability(self, logprobs):
        values = {i: logprobs[f"token_id:{i}"] for i in self.candidates["yes"] +
                  self.candidates["no"]}
        if not all(math.isfinite(v) for v in values.values()):
            return None
        top = max(values.values())
        yes = sum(math.exp(values[i] - top) for i in self.candidates["yes"])
        no = sum(math.exp(values[i] - top) for i in self.candidates["no"])
        return yes / (yes + no)

    def __call__(self, bodies, labels):
        import httpx
        if self.client is None:
            timeout = 1 / self.runtime["min_output_tokens_per_second"] + \
                self.runtime["request_overhead_seconds"]
            self.client = httpx.Client(timeout=timeout, trust_env=False)
        probabilities, usage, model = [], {"input_tokens": 0, "output_tokens": 0}, None
        for body in bodies:
            response = self.client.post(self.endpoint + "/v1/completions", json=body)
            response.raise_for_status()
            raw = response.json()
            model = raw.get("model")
            for key, name in (("input_tokens", "prompt_tokens"),
                              ("output_tokens", "completion_tokens")):
                usage[key] += (raw.get("usage") or {}).get(name) or 0
            try:
                positions = raw["choices"][0]["logprobs"]["top_logprobs"]
                p = self.probability(positions[0]) if len(positions) == 1 else None
            except (KeyError, IndexError, TypeError):
                p = None
            probabilities.append(p)
        return from_probabilities(probabilities, usage, model, requests=len(bodies))


def build_requests(predictor, case, text, language, labels, ja_names, prompt, shots):
    """Request body (or bodies) for one case; also used by --dry-run."""
    if isinstance(predictor, (OpenAIGeneration, LocalGeneration)):
        messages = generation_messages(case, text, language, labels, ja_names, prompt, shots)
        if isinstance(predictor, LocalGeneration):
            return predictor.request(messages, labels, case["case_id"])
        return predictor.request(messages, labels)
    state = report_text(case, text, prompt["report_headings"])
    questions = {x: question(x, language, prompt, ja_names) for x in labels}
    if isinstance(predictor, TypedQwen):
        return [predictor.request(state + "\n\n" + questions[x] + "\n"
                                  + prompt["typed_output"][language]) for x in labels]
    return predictor.request(state, questions)
