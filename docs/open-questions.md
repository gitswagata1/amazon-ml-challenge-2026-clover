# Open questions (what is not in the files yet)

Everything in these docs comes from the team's two repositories, the problem statement, and our own profile of the
submitted `matching_results.tsv`. These items could not be verified and are marked `[not in the files]` where they
appear.

| # | Missing | Why it matters | Where it would come from |
|---|---|---|---|
| 1 | **Private leaderboard score and final rank** | The final ranking used the private split; several French choices were tuned on the public split | Competition portal / Unstop results page |
| 2 | **Exact "Top 500" rank** | The README states Top 500 as reported by the team; the logs only record ~1,069th after v7 | Results announcement (2 Oct 2026) |
| 3 | **The last three uploads (0.986525 → 0.986691)** | They appear in the final package README but not in [`logs/submissions.md`](../logs/submissions.md), which ends at 0.986371 | Portal history or the team |
| 4 | **Is the profiled `matching_results.tsv` the 0.986691 file?** | [the data guide](data.md)'s profile (5,797,017 links) is of the file provided to us; the logs don't record the final file's link count | A checksum of the uploaded file |
| 5 | **Country breakdown of the submission** | Needs `test_source1.tsv` to join countries | The competition data (not redistributable) |
| 6 | **`candidate_pairs.tsv` and blocking statistics on test** | Candidate-set size was a ranking factor; the docs quote ~25 per S1 / 43.9 M pairs from the team's documentation | The final submission zip |
| 7 | **End-to-end runtime and GPU-hours** | Useful for anyone reproducing | Logs not committed (`logs/` is git-ignored) |
| 8 | **The challenge walkthrough video** | It was shared as a reference but not reviewed for these docs | The organisers' video |

If you can fill any of these in, please open an issue or a pull request.

---

<sub>← [Debrief script](debrief-script.md) · [Home](../README.md)</sub>
