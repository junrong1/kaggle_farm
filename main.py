import webbrowser
from pathlib import Path

from kaggle_environments import make

env = make("kaggriculture", debug=True)
env.run(["random", "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Text render straight to the terminal
print(env.render(mode="ansi"))

# Interactive render: write the player HTML out and open it in the browser
out = Path("render.html")
out.write_text(env.render(mode="html", width=800, height=600), encoding="utf-8")
print(f"Wrote {out.resolve()}")
webbrowser.open(out.resolve().as_uri())
