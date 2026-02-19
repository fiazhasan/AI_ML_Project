# ML_project - Push to GitHub
# Is script ko tab chalao jab GitHub par repo create ho chuka ho

Write-Host "Pushing ML_project to GitHub..." -ForegroundColor Cyan
Set-Location $PSScriptRoot

# Push main branch
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDone! Project successfully pushed to GitHub." -ForegroundColor Green
    Write-Host "Repo: https://github.com/fiazhasan/ML_project" -ForegroundColor Yellow
} else {
    Write-Host "`nPush failed. Make sure:" -ForegroundColor Red
    Write-Host "  1. GitHub par 'ML_project' name se naya repo create kiya ho"
    Write-Host "  2. 'Add README' checkbox OFF ho"
    Write-Host "  3. Browser mein GitHub login ho"
}
