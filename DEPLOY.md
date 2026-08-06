# Deploying the walkthrough to Streamlit Community Cloud

Free, no card, and the app needs no API key to be fully usable.

## 1. Commit the app and its samples

```powershell
cd D:\dev\projects\pecos-credit-intelligence
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

python scripts\export_samples.py     # regenerate from YOUR 12-deal corpus
git add app docs/samples
git status --short
```

**Check `docs/samples/` is actually staged.** If `.gitignore` has a broad
`docs/` or `*.json` rule it will be skipped, the deploy will succeed, and every
page will say "sample not present". Force it if needed:

```powershell
git add -f docs/samples
```

```powershell
git commit -m "feat: streamlit walkthrough and committed sample artefacts"
git push
```

## 2. Deploy

Go to **share.streamlit.io** → sign in with GitHub → **Create app** →
**Deploy a public app from GitHub**.

| field | value |
|---|---|
| Repository | `vamkotss/pecos-credit-intelligence` |
| Branch | `main` |
| **Main file path** | `app/streamlit_app.py` |
| App URL | `pecos-credit-intelligence` |

**Forward slashes.** Streamlit Cloud runs Linux; `app\streamlit_app.py` is the
error you hit before.

Then open **Advanced settings**:

| field | value |
|---|---|
| Python version | 3.12 |
| Requirements file | `app/requirements.txt` |

That last one matters. Left at the repository's `requirements.txt`, the build
installs ReportLab, PyMuPDF, pdfplumber and pytesseract to render some tables,
and takes several minutes longer for nothing.

Click **Deploy**. First build takes two or three minutes.

## 3. Optional: enable the Claude answering path

Without a key the app works fully, using the extractive generator. With one, the
"Ask the loan file" page offers Claude as well.

In the app's **⋮ → Settings → Secrets**, paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Streamlit secrets are encrypted and never appear in the repository. **The
repository's own CI secret scan would fail the build if a key were committed**,
which is that check working.

Judge whether it is worth it. A public app with a key is a public app spending
your money, and the extractive path demonstrates citation validation, grounding
and refusal on its own. Deploy without it first.

## 4. Check it

Click every page. The two that would fail first are **Ingestion** and **Memo
agent** — both index into sample files, so a missing `docs/samples/` shows there
before anywhere else.

Then add the link to the top of the README, under the CI badge:

```markdown
[![CI](https://github.com/vamkotss/pecos-credit-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/vamkotss/pecos-credit-intelligence/actions/workflows/ci.yml)

**[Live walkthrough →](https://pecos-credit-intelligence.streamlit.app)**
```

## Notes

**Apps sleep after inactivity** and take ~30 seconds to wake. Fine for a
portfolio link; worth clicking it an hour before an interview.

**Redeploys happen on push.** Anything committed to `main` that breaks the app
breaks the public link, so run it locally before pushing changes to
`app/` or `docs/samples/`.

**The chunk index is two deals, not twelve.** 58 KB of real chunks is enough to
demonstrate retrieval; the whole corpus would be several megabytes of text
nobody reads. `export_samples.py` controls which deals go in.
