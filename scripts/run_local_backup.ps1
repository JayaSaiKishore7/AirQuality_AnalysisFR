<#
.SYNOPSIS
  Free backup runner for the daily pipeline on a Windows PC.

.DESCRIPTION
  Every evening, if the live page does not have today's forecast yet (GitHub Actions did not
  run, e.g. outage or billing lock), runs scripts/daily_pipeline.py here and publishes it.
  If GitHub already published, it does nothing.

  It works in its own clone (%LOCALAPPDATA%\AirQualityRunner\repo, reset to origin/main on
  every run) and its own virtualenv, never in your working copy. It pushes with a GitHub
  deploy key (%USERPROFILE%\.ssh\aq_deploy_key) and deploys with your `hf auth login` token.
  Windows notifications: when it had to publish, and when anything fails.

.EXAMPLE
  .\scripts\run_local_backup.ps1 -Setup        # one-time: clone, venv, deploy key, daily task at 20:00
  .\scripts\run_local_backup.ps1               # run now (what the task does)
  .\scripts\run_local_backup.ps1 -Unregister   # remove the daily task
#>
param(
    [switch]$Setup,
    [switch]$Unregister,
    [string]$At = "20:00",
    [string]$Space = "JayaSaiKishore/airquality-paca",
    [string]$GitHubRepo = "JayaSaiKishore7/AirQuality_AnalysisFR"
)

$ErrorActionPreference = "Stop"
$TaskName = "AirQuality daily backup"
$Runner = Join-Path $env:LOCALAPPDATA "AirQualityRunner"
$Repo = Join-Path $Runner "repo"
$Py = Join-Path $Runner "venv\Scripts\python.exe"
$Key = Join-Path $env:USERPROFILE ".ssh\aq_deploy_key"
$Log = Join-Path $Runner "backup.log"
$PageUrl = "https://huggingface.co/spaces/$Space"

function Write-Log([string]$Text) {
    Add-Content -Encoding utf8 $Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Text"
}

function Show-Toast([string]$Title, [string]$Message) {
    try {
        $esc = { param($s) [Security.SecurityElement]::Escape($s) }
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml("<toast activationType='protocol' launch='$(& $esc $PageUrl)'><visual><binding template='ToastGeneric'>" +
            "<text>$(& $esc $Title)</text><text>$(& $esc $Message)</text></binding></visual></toast>")
        $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show(
            [Windows.UI.Notifications.ToastNotification]::new($xml))
    } catch { Write-Log "toast failed: $_" }
}

function Install-Dependencies {
    # reinstall only when the pinned versions change
    $stamp = Join-Path $Runner "deps.sha"
    $hash = (Get-FileHash (Join-Path $Repo "constraints.txt")).Hash + (Get-FileHash (Join-Path $Repo "requirements-pipeline.txt")).Hash
    if ((Test-Path $stamp) -and (Get-Content $stamp) -eq $hash) { return }
    & $Py -m pip install --quiet --disable-pip-version-check -r (Join-Path $Repo "requirements-pipeline.txt") `
        huggingface_hub==2.4.0 -c (Join-Path $Repo "constraints.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Set-Content $stamp $hash
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed scheduled task '$TaskName'"
    return
}

if ($Setup) {
    New-Item -ItemType Directory -Force $Runner | Out-Null
    if (-not (Test-Path (Join-Path $Repo ".git"))) {
        git clone --quiet "https://github.com/$GitHubRepo.git" $Repo
    }
    if (-not (Test-Path $Py)) {
        $base = (Get-Command py -ErrorAction SilentlyContinue)
        if ($base) { & py -3.13 -m venv (Join-Path $Runner "venv") } else { & python -m venv (Join-Path $Runner "venv") }
    }
    Write-Output "Installing dependencies into $Runner\venv (first time takes a few minutes)..."
    Install-Dependencies

    if (-not (Test-Path $Key)) {
        New-Item -ItemType Directory -Force (Split-Path $Key) | Out-Null
        ssh-keygen -q -t ed25519 -N '""' -C "airquality-backup@$env:COMPUTERNAME" -f $Key
    }
    $hasToken = (& $Py -c "from huggingface_hub import get_token; print(bool(get_token()))").Trim()

    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Repo\scripts\run_local_backup.ps1`""
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "Publishes the PACA air-quality forecast if GitHub Actions has not today" -Force | Out-Null
    # this task also alerts on failures, so the separate page watcher is no longer needed
    if (Get-ScheduledTask -TaskName "AirQuality live page check" -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName "AirQuality live page check" -Confirm:$false
    }

    Write-Output ""
    Write-Output "Scheduled '$TaskName' daily at $At (runs at next logon if the PC was off). Log: $Log"
    Write-Output ""
    Write-Output "One-time steps left:"
    Write-Output "1. Add this deploy key at https://github.com/$GitHubRepo/settings/keys/new"
    Write-Output "   Title: airquality-backup   [x] Allow write access"
    Write-Output "   $(Get-Content "$Key.pub")"
    if ($hasToken -ne "True") {
        Write-Output "2. Log in to Hugging Face (a token with write access to the Space):"
        Write-Output "   & `"$Runner\venv\Scripts\hf.exe`" auth login"
    } else {
        Write-Output "2. Hugging Face login: already done"
    }
    return
}

# ---- daily run ----
New-Item -ItemType Directory -Force $Runner | Out-Null
try {
    if (-not (Test-Path $Py)) { throw "Runner not set up. Run: .\scripts\run_local_backup.ps1 -Setup" }
    git -C $Repo fetch --quiet origin
    git -C $Repo reset --quiet --hard origin/main
    git -C $Repo clean -fdq
    Install-Dependencies

    $env:HF_SPACE_ID = $Space
    $env:GITHUB_REPO = $GitHubRepo
    $env:DEPLOY_KEY_FILE = $Key
    $env:AQ_RUNNER = "the backup runner on $env:COMPUTERNAME"
    Push-Location $Repo
    # the pipeline logs to stderr; under "Stop", PowerShell 5.1 would treat that as a fatal error
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Py scripts/daily_pipeline.py --skip-if-published-today 2>&1 | ForEach-Object { "$_" } | Out-String
        $code = $LASTEXITCODE
    } finally { Pop-Location; $ErrorActionPreference = "Stop" }
    Set-Content -Encoding utf8 (Join-Path $Runner "last-run.txt") $output

    if ($code -ne 0) {
        $why = ($output -split "\r?\n" | Where-Object { $_ -match "^FAILED at step" } | Select-Object -Last 1)
        Write-Log "FAILED (exit $code): $why"
        Show-Toast "Air-quality backup run failed" "$why  Details: $Runner\last-run.txt"
        exit $code
    }
    if ($output -match "already published") {
        Write-Log "GitHub already published today; nothing to do"
    } else {
        Write-Log "Published today's forecast from this PC (GitHub had not)"
        Show-Toast "Air-quality forecast published from this PC" "GitHub Actions had not published today. Check its status."
    }
} catch {
    Write-Log "ERROR: $_"
    Show-Toast "Air-quality backup run failed" "$_"
    exit 1
}
