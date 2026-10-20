# Releasing v2.0.0

1. **Merge** the pull request `design/v2.0-detect-respond` → `main` once CI is green.
2. **Release:** tag `v2.0.0` on `main`, title "v2.0.0 — detection and response around the decision test". Zenodo archives it under the concept DOI 10.5281/zenodo.23326083.
3. **Close the loop:** once Zenodo has minted the version DOI, add it to `CITATION.cff` (`doi:` and `identifiers`) and to the README's Citation section. Do not cut another release for this.

The technical report (`docs/report/`) still covers v1.0–v1.1; v1.2–v2.0 are documented in `docs/design/`.

---

# Releasing v1.1.0

v1.0.0 is already public, so only these steps remain.

1. **Merge** pull request #1 (`design/v1.1-decoys` → `main`) once CI is green.
2. **Check the date** in `CITATION.cff` (`date-released`) and correct it if the release happens on another day.
3. **Zenodo:** if the toggle was enabled for v1.0.0, nothing to do; Zenodo archives every new release under the same concept DOI. If it was not, enable it *now*, before step 4.
4. **Release:** *Releases → Draft a new release*, tag `v1.1.0` on `main`, title "v1.1.0 — decoy discrimination and a calibrated decision", attach `docs/report/report.pdf`.
5. Put the new **version DOI** into the README's Citation section.

Everything below is the original v1.0.0 checklist, kept for reference.

---

# Releasing v1.0.0

A one-time checklist, from a local folder to a citable DOI.

## 0. Before anything is public

- [ ] **IP.** Confirm that the code was not written under a HiWi or employment contract, and that no university, incubator or other agreement claims it. If in doubt, ask before publishing; a public repository cannot be un-published in practice.
- [ ] **Placeholders.** Find every placeholder:
  ```bash
  grep -rn "GIVEN-NAME\|FAMILY-NAME\|YOUR-USERNAME" --exclude-dir=.git .
  ```
  Replace them with your real values in these files:
  - `CITATION.cff`
  - `.zenodo.json`
  - `docs/report/report.tex` (`\authorname`, `\repourl`)
  - `NOTICE` and `pyproject.toml` (the copyright holder and author, currently `d3st`)
- [ ] **Rebuild the report** (the `.pdf` is committed so readers need no LaTeX):
  ```bash
  cd docs/report && pdflatex report.tex && pdflatex report.tex && cd -
  ```
- [ ] **Re-run the tests:** `pytest -q`

If you change any code, re-run all four experiment scripts before release:
```bash
python experiments/run_benchmark.py --seeds 100
python experiments/calibrate_prediction.py --seeds 50
python experiments/ablation_prior_tempering.py --seeds 100
python experiments/make_media.py
python experiments/make_report_tables.py
```
Then update the README numbers from `results/summary.md` and rebuild the report.

## 1. GitHub

```bash
git remote add origin https://github.com/3cthulhu3/argus-autonomy.git
git push -u origin main
```
Then:
- On the *Actions* tab, wait for CI to go green.
- In *About*, add a one-line description and the topics `wildfire`, `uav`, `active-perception`, `informative-path-planning`.

## 2. Zenodo, which must come *before* the GitHub release

1. Log in at zenodo.org with your GitHub account.
2. Under *Settings → GitHub*, flip the toggle for `argus-autonomy`.

Zenodo only archives releases that are made *after* the toggle is on.

## 3. Release

On GitHub, go to *Releases → Draft a new release*:
- tag `v1.0.0`;
- title "v1.0.0 — first public release";
- attach `docs/report/report.pdf`.

Within a few minutes Zenodo mints two DOIs:
- the **version DOI**, which identifies v1.0.0 specifically. Cite this one.
- the **concept DOI**, which always resolves to the latest version.

## 4. Close the loop

- [ ] Add the DOI badge Zenodo shows you to the top of `README.md`.
- [ ] Add `doi: 10.5281/zenodo.XXXXXXX` to `CITATION.cff`, then commit and push. Do not cut another release for this.
- [ ] Open the repository in a private browser window and check three things:
  - the GIF renders;
  - the report link works;
  - the commands in *Run it* work from a fresh clone.
