# Steps to Push This Project to GitHub

## What’s Already Done
- ✅ Git initialized
- ✅ First commit on the `main` branch
- ✅ Branch name: `main`

---

## Step 1: Create a New Repository on GitHub

1. Open **https://github.com/new**
2. **Repository name:** `ML_project` (or any name you prefer)
3. Select **Public**
4. **Do not** check “Add a README” (this project already has a README)
5. Click **Create repository**

---

## Step 2: Add Remote and Push

After creating the repo, GitHub will show you a URL. Use it with these commands (replace with your username and repo name):

```powershell
cd "c:\Users\Fiaz\Desktop\ML_project"

# Add remote (put your username and repo name in the URL)
git remote add origin https://github.com/YOUR_USERNAME/ML_project.git

# Push main branch
git push -u origin main
```

**Using SSH instead:**
```powershell
git remote add origin git@github.com:YOUR_USERNAME/ML_project.git
git push -u origin main
```

---

## Step 3: Create a New Branch (Optional)

To create a new branch and work on it:

```powershell
# Create a new branch (e.g. dev)
git checkout -b dev

# Make changes, then:
git add .
git commit -m "Your message"
git push -u origin dev
```

**Switch back to main later:**
```powershell
git checkout main
```

**List all branches:**
```powershell
git branch -a
```

---

## Useful Git Commands (Quick Reference)

| Command | Description |
|--------|-------------|
| `git status` | See which files have changed |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Create a commit |
| `git push` | Push to GitHub |
| `git pull` | Pull latest from GitHub |
| `git branch` | List branches |
| `git checkout -b branch-name` | Create and switch to a new branch |
| `git log --oneline` | View commit history |

---

## If You Already Added a Remote

If you added the wrong URL by mistake:

```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/ML_project.git
git push -u origin main
```
