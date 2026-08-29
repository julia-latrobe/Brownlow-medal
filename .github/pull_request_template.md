## What does this change?

<!-- A sentence or two. What is different after this is merged? -->

## Why?

<!-- The problem this solves, or the question it answers. Link an issue if there is one. -->

## How was it checked?

- [ ] `pytest` passes locally
- [ ] `ruff check .` is clean
- [ ] I added or updated tests for the behaviour I changed

## Does this change the model's output?

<!-- Delete this section if not. Otherwise paste the holdout metrics before and
     after, so a reviewer can see whether the model got better or worse. -->

| Metric | Before | After |
| --- | --- | --- |
| Top-1 accuracy | | |
| Top-3 recall | | |
| Spearman | | |

- [ ] I regenerated `docs/index.html` with `brownlow report` and committed it
