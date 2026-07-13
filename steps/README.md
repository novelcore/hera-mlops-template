# steps/ — your pipeline step code

One directory per pipeline step: your code, plus a Dockerfile in the
real repo (the PoC keeps only minimal entry points).

Each step receives the fully resolved experiment config (`--params`)
and should read only the config sections it declares via `reads=` in
`pipeline.py` — that declaration is checked at release time, so a
renamed or deleted section fails loudly instead of breaking your step
mid-run.
