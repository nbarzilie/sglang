# SGLang Contribution Guide: Nixl Testing Suite

This document outlines the Git configuration and workflow for developing the Nixl backend testing suite for `sgl-project/sglang`.

---

## 1. Repository Architecture

* **Upstream (The Official Repo):** `https://github.com/sgl-project/sglang`
* **Origin (Your Personal Fork):** `https://github.com/nbarzilie/sglang`
* **Feature Branch:** `feature/nixl-testing-suite`

---

## 2. CLI Configuration Verification

To verify your local repository is correctly mapped to both your personal fork and the official repository, run:

```bash
git remote -v

```

**Expected Output:**

```text
origin    [https://github.com/nbarzilie/sglang.git](https://github.com/nbarzilie/sglang.git) (fetch)
origin    [https://github.com/nbarzilie/sglang.git](https://github.com/nbarzilie/sglang.git) (push)
upstream  [https://github.com/sgl-project/sglang.git](https://github.com/sgl-project/sglang.git) (fetch)
upstream  [https://github.com/sgl-project/sglang.git](https://github.com/sgl-project/sglang.git) (push)

```

To see your current active branch and tracking status:

```bash
git branch -vv

```

---

## 3. Daily Workflow Commands

### Save & Backup Progress (To your Fork)

Run these commands to commit your local testing modifications and back them up safely online:

```bash
# 1. Stage all new test files and modifications
git add .

# 2. Commit changes using conventional commit style
git commit -m "test(nixl): add unit and functional testing suite updates"

# 3. Push to your fork (The '-u' flag is only required on the very first push)
git push origin feature/nixl-testing-suite

```

### Keep Your Branch Updated (With SGLang Main)

Because SGLang is highly active, use a **rebase workflow** to bring in upstream changes without cluttering your testing history with merge commits.

```bash
# 1. Ensure your current working directory is clean
git status

# 2. Fetch the latest changes from the official repo
git fetch upstream main

# 3. Rebase your test commits cleanly on top of the updated main branch
git rebase upstream/main

# 4. Force-push to update your remote fork backup after a rebase
git push --force-with-lease origin feature/nixl-testing-suite

```

> ⚠️ **Note on Rebasing:** If Git pauses due to a conflict during rebase, open the affected file, resolve the conflicting lines, and run:
> ```bash
> git add <resolved-file>
> git rebase --continue
> 
> ```
> 
> 

```