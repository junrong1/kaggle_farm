"""Log kaggriculture episodes to disk as training data.

Two formats are written per episode:
  data/selfplay/<id>.jsonl.gz  -- one JSON line per (step, player) transition
  data/selfplay/<id>.meta.json -- seed, config, final rewards, agent names

Because the env is deterministic given (seed, actions), the meta file alone is
enough to regenerate every observation later via replay_actions().
"""

from __future__ import annotations

import gzip
import json
import uuid
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make

OUT_DIR = Path("data/selfplay")


def _to_plain(x: Any) -> Any:
    """kaggle_environments hands back Struct objects; make them JSON-safe."""
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x


def _record_agent(inner: Callable, player: int, sink: list) -> Callable:
    """Wrap an agent so every (obs, action) it actually saw is captured.

    Recording at decision time avoids the off-by-one in env.steps, where
    steps[t].action is the action chosen from steps[t-1].observation.
    """

    def wrapped(observation, configuration):  # two named params: arity matters
        obs = _to_plain(observation)
        action = inner(observation) if inner.__code__.co_argcount == 1 else inner(observation, configuration)
        sink.append(
            {
                "step": obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0)),
                "day": obs.get("day"),
                "hour": obs.get("hour"),
                "player": player,
                "money": obs["farms"][player]["money"] if obs.get("farms") else None,
                "obs": obs,
                "action": _to_plain(action),
            }
        )
        return action

    return wrapped


def log_episode(agents: list[str | Callable], seed: int | None = None, out_dir: Path = OUT_DIR) -> dict:
    """Run one episode and write it to disk. Returns the meta dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_id = uuid.uuid4().hex[:12]

    config = {"seed": seed} if seed is not None else {}
    env = make("kaggriculture", configuration=config, debug=False)

    rows: list[dict] = []
    runnable = [
        _record_agent(env.agents[a] if isinstance(a, str) else a, i, rows)
        for i, a in enumerate(agents)
    ]
    env.run(runnable)

    final = env.steps[-1]
    rewards = [s.reward for s in final]

    jsonl_path = out_dir / f"{ep_id}.jsonl.gz"
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as f:
        for r in rows:
            r["final_reward"] = rewards[r["player"]]
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    meta = {
        "id": ep_id,
        "seed": env.info.get("seed"),
        "agents": [a if isinstance(a, str) else getattr(a, "__name__", "callable") for a in agents],
        "rewards": rewards,
        "statuses": [s.status for s in final],
        "steps": len(env.steps),
        "configuration": _to_plain(env.configuration),
        "actions": [[r["action"] for r in rows if r["player"] == p] for p in range(len(agents))],
    }
    (out_dir / f"{ep_id}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def replay_actions(meta_path: Path):
    """Rebuild an episode's full state from seed + logged actions."""
    meta = json.loads(Path(meta_path).read_text())
    env = make("kaggriculture", configuration={"seed": meta["seed"]}, debug=False)
    def replayer(recorded: list) -> Callable:
        # Must be a genuine closure with exactly one parameter: kaggle_environments
        # truncates the call args to the agent's co_argcount, so a default-arg
        # lambda (obs, _q=q) gets handed the *configuration* as _q.
        queue = list(recorded)

        def agent(observation):
            return queue.pop(0) if queue else {"farmer": ["PASS"], "hands": [], "market": []}

        return agent

    env.run([replayer(a) for a in meta["actions"]])
    return env


if __name__ == "__main__":
    m = log_episode(["random", "random"], seed=42)
    print(json.dumps({k: v for k, v in m.items() if k != "actions"}, indent=1))
