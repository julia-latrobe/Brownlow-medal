# Getting started

This guide takes you from a computer with nothing installed to running the
model, changing it, and getting your change published. It assumes you can write
Python but have never used VS Code or GitHub before. Nothing here is guessy —
follow it top to bottom and it will work.

If you already have Git, Python and VS Code set up, skip to
[Step 4](#step-4-get-the-code-onto-your-computer).

**Contents**

1. [What this project is](#what-this-project-is)
2. [Step 1: Install the three tools](#step-1-install-the-three-tools)
3. [Step 2: Tell Git who you are](#step-2-tell-git-who-you-are)
4. [Step 3: Connect VS Code to GitHub](#step-3-connect-vs-code-to-github)
5. [Step 4: Get the code onto your computer](#step-4-get-the-code-onto-your-computer)
6. [Step 5: Set up Python for this project](#step-5-set-up-python-for-this-project)
7. [Step 6: Check it works](#step-6-check-it-works)
8. [Step 7: Run the model](#step-7-run-the-model)
9. [The everyday workflow](#the-everyday-workflow)
10. [Things to try](#things-to-try)
11. [When something goes wrong](#when-something-goes-wrong)
12. [Glossary](#glossary)

---

## What this project is

A model that predicts AFL Brownlow Medal votes. Umpires award 3, 2 and 1 votes
after every home-and-away match; this project learns which performances earn
those votes and projects the season's total for every player.

Running it produces a results page — charts and tables of who is projected to
poll — which gets published to the web.

The modelling detail is in [README.md](README.md). This file is only about
getting set up.

---

## Step 1: Install the three tools

You need Git (tracks changes), Python (runs the model) and VS Code (where you
edit). Install all three, then **restart your computer** — this matters, because
installers add things to your system PATH and already-open programs won't see
them until they restart.

### Git

- **Windows:** download from [git-scm.com/download/win](https://git-scm.com/download/win).
  Run the installer and accept every default. There are a lot of screens; the
  defaults are fine on all of them.
- **macOS:** open Terminal (press `Cmd+Space`, type "Terminal", press Enter) and
  run `git --version`. If Git isn't installed, macOS offers to install it — say
  yes. Otherwise install from [git-scm.com/download/mac](https://git-scm.com/download/mac).

### Python

Download **Python 3.11 or newer** from [python.org/downloads](https://www.python.org/downloads/).

> **Windows — this one matters.** On the first screen of the installer, tick
> **"Add python.exe to PATH"** at the bottom *before* clicking Install. It is easy
> to miss, and skipping it causes most of the "python is not recognized" problems
> people hit later.

### VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/) and install
it with the defaults.

### Check they all worked

Restart your computer, then open a terminal:

- **Windows:** press the Start button, type `powershell`, press Enter.
- **macOS:** press `Cmd+Space`, type `terminal`, press Enter.

Run these three commands. Each should print a version number:

```bash
git --version
python --version     # on macOS, try python3 --version
code --version
```

If any of them says "not recognized" or "command not found", see
[When something goes wrong](#when-something-goes-wrong).

---

## Step 2: Tell Git who you are

Git stamps your name and email on every change you make, so it needs to know
them once. Use the email address on your GitHub account.

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 3: Connect VS Code to GitHub

1. Open VS Code.
2. Click the **account icon** at the very bottom of the left-hand sidebar (a
   person outline, below the gear icon).
3. Choose **Sign in to Sync Settings** or **Sign in with GitHub**.
4. Your browser opens and asks you to authorise VS Code. Say yes.
5. The browser hands you back to VS Code. The account icon now shows your
   GitHub username.

You only ever do this once per computer. From here on, VS Code can clone
repositories and push your work without asking for a password.

---

## Step 4: Get the code onto your computer

"Cloning" means downloading a copy of the project that stays connected to the
original, so you can send your changes back.

1. In VS Code, press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS). This opens the
   **Command Palette** — a search box for every VS Code command. You will use it
   constantly.
2. Type `Git: Clone` and press Enter.
3. Paste the repository URL:

   ```
   https://github.com/julia-latrobe/Brownlow-medal.git
   ```

4. Choose a folder to put it in — your Documents folder is fine.
5. When VS Code asks *"Would you like to open the cloned repository?"*, click
   **Open**.

You should now see the project's files in the sidebar on the left.

VS Code will probably show a notification offering to install the recommended
extensions. **Click Install.** The project lists the ones it expects — Python
support, the linter, and GitHub integration — in `.vscode/extensions.json`.

---

## Step 5: Set up Python for this project

A **virtual environment** is a private Python installation that belongs to this
project alone. It keeps this project's packages from colliding with anything else
on your machine. Every Python project should have one.

Open a terminal *inside VS Code*: menu **Terminal → New Terminal**, or press
`` Ctrl+` `` (the backtick key, above Tab). It opens at the project folder
already, which saves you navigating there.

Create the environment:

```bash
# Windows
python -m venv .venv

# macOS
python3 -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. That is how you know it is active. **You
need to activate it every time you open a new terminal** — VS Code usually does
it for you once it knows about the environment.

> **Windows:** if activating gives you a red error about "running scripts is
> disabled on this system", run this once and try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

Now install the project and its tools:

```bash
pip install -e ".[data,dev]"
```

The `-e` means "editable": Python reads the code straight from your folder, so
when you edit a file the change takes effect immediately with no reinstalling.

Finally, tell VS Code to use this environment: press `Ctrl+Shift+P`, type
`Python: Select Interpreter`, and choose the one with `.venv` in its path
(it is usually labelled "Recommended").

---

## Step 6: Check it works

Run the tests:

```bash
pytest -q
```

You should see a row of dots and no failures. The tests build their own fake data
so this works before you have downloaded anything.

You can also run them from VS Code's **Testing** panel — the flask icon in the
left sidebar. It lists every test, and clicking one runs just that test. Handy
when you are trying to fix a single failure.

---

## Step 7: Run the model

Download the real data (about 14 MB; it is cached, so this is a one-off):

```bash
brownlow fetch
```

Train on past seasons and check the model against a season it has never seen:

```bash
brownlow backtest --train 2015-2023 --test 2024-2025 --compare
```

Then project the current season and build the results page:

```bash
brownlow predict --seasons 2015-2026 --train 2015-2025 --test 2025 --predict 2026
```

That writes `docs/index.html`. To look at it, right-click the file in the VS Code
sidebar and choose **Reveal in File Explorer** (or **Reveal in Finder**), then
double-click it to open in your browser.

If you would rather not touch the terminal, press **F5** in VS Code and pick one
of the ready-made run configurations from the dropdown.

---

## The everyday workflow

This is the loop you will repeat for every change. It is worth doing it the same
way every time until it is automatic.

### 1. Start from an up-to-date main

`main` is the official version of the project. Before starting anything new, get
the latest:

```bash
git checkout main
git pull
```

### 2. Make a branch

A **branch** is your own copy of the project to work on, so half-finished work
never disturbs anyone else.

```bash
git checkout -b add-last-quarter-goals
```

Name it after what you are doing. In VS Code you can also click the branch name
in the bottom-left status bar and choose **Create new branch**.

### 3. Change something, and check it

Edit files. Then run the tests and the linter:

```bash
pytest -q
ruff check .
```

If your change affects the model, rerun the backtest and see whether the numbers
moved in the right direction.

### 4. Commit

A **commit** is a save point with a message explaining the change. In VS Code:

1. Click the **Source Control** icon in the left sidebar (the branching-arrows
   symbol), or press `Ctrl+Shift+G`.
2. You will see your changed files. Hover over a file and click **+** to *stage*
   it — that means "include this in the next commit". Stage everything that
   belongs to this change, and nothing that does not.
3. Type a message in the box at the top. Write what changed and why:
   *"Add last-quarter goals as a feature"*, not *"update"*.
4. Click **Commit**.

The equivalent in the terminal:

```bash
git add src/brownlow/features.py
git commit -m "Add last-quarter goals as a feature"
```

Commit often. Small commits with clear messages are much easier to review, and
much easier to undo when one of them turns out to be wrong.

### 5. Push

Your commits are still only on your computer. **Pushing** sends them to GitHub:

```bash
git push -u origin add-last-quarter-goals
```

Or click **Publish Branch** in the Source Control panel.

### 6. Open a pull request

A **pull request** (PR) proposes merging your branch into `main`. It is where
review happens.

In VS Code, with the GitHub Pull Requests extension installed:

1. Click the **GitHub** icon in the left sidebar.
2. Click **Create Pull Request**.
3. Fill in the title and description. The description is pre-filled with a
   template — fill in the sections; they are the questions a reviewer will ask
   anyway.
4. Click **Create**.

You can also do it on github.com: after pushing, the repository page shows a
banner offering to open a pull request from your branch.

### 7. Watch the checks

Opening a PR automatically runs the tests on GitHub's machines. Look for the
checks section at the bottom of the PR:

- **Green tick** — tests passed. Ready for review.
- **Red cross** — something failed. Click **Details** to read the log and find
  out what. Push a fix to the same branch and the checks re-run by themselves.

### 8. Review, then merge

Someone reads the change and either approves it or asks for changes. Answer the
comments, push any fixes, and once it is approved click **Merge pull request**.

Then delete the branch — GitHub offers a button — and go back to step 1.

### 9. The site updates

If your change included a regenerated `docs/index.html`, merging it to `main`
triggers the **Publish site** workflow, which puts the new page on the web at:

```
https://julia-latrobe.github.io/Brownlow-medal/
```

Give it a minute or two. You can watch it happen in the repository's **Actions**
tab.

> One-time setup by the repository owner: **Settings → Pages → Source: GitHub
> Actions**. Until that is done the workflow will fail with a permissions error.

---

## Things to try

Working examples to get the workflow into your hands. They get harder as they go.

### 1. Change some words

The gentlest possible round trip. Fix a typo, or improve a sentence in this file.
Branch, edit, commit, push, open a PR, merge. No code, no tests to break — just
the loop itself.

### 2. Add yourself to the contributors list

Open `CONTRIBUTING.md`, add your name at the bottom, and take it through the same
loop. This time notice what the diff looks like in the PR: GitHub shows exactly
which lines you added.

### 3. Change a setting and see the model move

Open `experiments/rank-model.json` and change `"alpha": 1.0` to `"alpha": 5.0`.
Alpha controls how strongly the model is regularised — higher means more cautious.

```bash
brownlow run experiments/rank-model.json --seasons 2015-2026
```

Did `top3_recall` go up or down? Put the before and after numbers in your PR
description. This is a real experiment with a real result, even though you only
changed one character.

### 4. Add a new experiment

Copy `experiments/rank-model.json` to a new file with a new `"name"`, change
something — the training seasons, the feature set, the model — and run it.

```bash
brownlow run experiments/my-idea.json --seasons 2015-2026
brownlow compare
```

Because each experiment is its own file, this adds a run rather than replacing
one, and the results page grows a new entry in its dropdown. Commit the config
and the regenerated `docs/index.html` together.

### 5. Add a feature to the model

Open `src/brownlow/features.py`. The `add_derived_stats` function builds combined
statistics out of the raw ones. Add another — perhaps `marks_inside_50` divided by
`inside_50s`, or contested possessions as a share of total disposals.

Then:
- add a test for it in `tests/test_features.py`
- run `pytest -q`
- run the backtest and see whether the metrics improved
- put the before and after numbers in your PR

This is the full loop: an idea, an implementation, a test, and evidence about
whether it helped.

### 6. Practise resolving a conflict

A **merge conflict** happens when two branches change the same lines. They look
alarming and are usually a two-minute fix, so it is worth meeting one on purpose
rather than in a hurry.

1. On `main`, edit a line in `README.md` and commit it.
2. Make a branch from an *earlier* point, edit the same line differently, commit.
3. Run `git merge main`. Git stops and marks the clash.
4. Open the file. VS Code highlights both versions with buttons above them:
   **Accept Current Change**, **Accept Incoming Change**, **Accept Both**.
5. Pick what the file should say, save, then `git add` the file and
   `git commit`.

Conflicts are Git refusing to guess. You decide, and it gets out of the way.

---

## When something goes wrong

**`python: command not found` or `'python' is not recognized`**
Python isn't on your PATH. On Windows, reinstall it and tick "Add python.exe to
PATH" on the first screen. On macOS, try `python3` instead of `python`.

**`pip: command not found`**
Your virtual environment probably isn't active — check for `(.venv)` at the start
of your prompt. If it isn't there, activate it (Step 5). Otherwise use
`python -m pip` instead of `pip`.

**`ModuleNotFoundError: No module named 'brownlow'`**
Either the environment isn't active, or the install step didn't run. Activate it
and run `pip install -e ".[data,dev]"` again.

**`ModuleNotFoundError: No module named 'pyreadr'`**
That package reads the data file and is optional. Install it with
`pip install "brownlow[data]"`.

**Windows: "running scripts is disabled on this system"**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate
again.

**VS Code can't find the tests, or imports show red squiggles**
It is pointing at the wrong Python. Press `Ctrl+Shift+P`,
`Python: Select Interpreter`, and pick the `.venv` one. Reload the window if the
squiggles persist (`Ctrl+Shift+P` → `Developer: Reload Window`).

**`git push` is rejected**
Someone else pushed first. Run `git pull` to bring their work in, resolve any
conflict, then push again.

**I committed to `main` by accident**
Nothing is lost. Make a branch at your current position, then move `main` back:

```bash
git branch my-work           # save your commits on a new branch
git reset --hard origin/main # put main back to matching GitHub
git checkout my-work         # carry on where you were
```

**I want to throw away my uncommitted changes to a file**

```bash
git restore path/to/file.py
```

**The tests fail and I don't understand why**
Run `pytest -q` and read the last few lines first — pytest prints the summary at
the bottom. Then run just the failing test for a fuller report:
`pytest tests/test_model.py::TestPlackettLuceModel::test_fits_and_predicts -v`.

---

## Glossary

| Term | What it means |
| --- | --- |
| **repository** (repo) | The project: its files plus the whole history of changes to them. |
| **clone** | Download a copy of a repository that stays linked to the original. |
| **branch** | An independent line of work, so unfinished changes don't disturb anyone. |
| **commit** | A save point, with a message explaining what changed and why. |
| **stage** | Mark which changes go into your next commit. |
| **push** | Send your commits from your computer up to GitHub. |
| **pull** | Bring other people's commits down from GitHub to your computer. |
| **pull request** (PR) | A proposal to merge your branch into `main`, where review happens. |
| **merge** | Combine one branch's changes into another. |
| **merge conflict** | Two branches changed the same lines; you decide which version wins. |
| **`main`** | The official, current version of the project. |
| **`origin`** | Git's name for the copy of the repo on GitHub. |
| **CI** | "Continuous integration" — the tests GitHub runs automatically on every PR. |
| **GitHub Actions** | The system that runs CI and publishes the site. |
| **virtual environment** | A private Python install for one project, kept in `.venv`. |

---

Once you are set up, [README.md](README.md) explains how the model works and what
each command does. [CONTRIBUTING.md](CONTRIBUTING.md) covers the conventions this
project follows.
