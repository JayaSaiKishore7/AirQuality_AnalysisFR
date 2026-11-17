<#
.SYNOPSIS
  Out-of-band check of the live forecast page, independent of GitHub Actions.

.DESCRIPTION
  Runs scripts/check_deployed_page.py against the Hugging Face Space. If the page is
  unreachable, broken or older than -MaxAgeHours, shows a Windows notification (click
  it to open the page). Every run is logged to %LOCALAPPDATA%\AirQualityWatch\watch.log.

  This catches what the in-Actions checks cannot: Actions itself not running
  (e.g. the November 2026 billing lock).

.EXAMPLE
  .\scripts\watch_live_page.ps1                 # check now
  .\scripts\watch_live_page.ps1 -Register       # check daily at 18:00 (and at next logon if missed)
  .\scripts\watch_live_page.ps1 -Unregister     # remove the daily check
#>
param(
    [string]$Url = "https://jayasaikishore-airquality-paca.static.hf.space/",
    [double]$MaxAgeHours = 36,
    [string]$At = "18:00",
    [switch]$Register,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$TaskName = "AirQuality live page check"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $env:LOCALAPPDATA "AirQualityWatch"
$Log = Join-Path $LogDir "watch.log"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed scheduled task '$TaskName'"
    return
}

if ($Register) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -Url `"$Url`" -MaxAgeHours $MaxAgeHours"
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    # StartWhenAvailable: if the PC was off at $At, run at the next opportunity
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Description "Alerts when the PACA air-quality forecast page is stale or broken ($Repo)" -Force | Out-Null
    Write-Output "Registered '$TaskName': daily at $At, log: $Log"
    return
}

function Show-Alert([string]$Title, [string]$Message) {
    $esc = { param($s) [Security.SecurityElement]::Escape($s) }
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml("<toast activationType='protocol' launch='$(& $esc $Url)' scenario='reminder'><visual><binding template='ToastGeneric'>" +
        "<text>$(& $esc $Title)</text><text>$(& $esc $Message)</text></binding></visual>" +
        "<actions><action content='Open page' activationType='protocol' arguments='$(& $esc $Url)'/></actions></toast>")
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show(
        [Windows.UI.Notifications.ToastNotification]::new($xml))
}

New-Item -ItemType Directory -Force $LogDir | Out-Null
$python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$output = & $python (Join-Path $Repo "scripts\check_deployed_page.py") --url $Url `
    --max-age-hours $MaxAgeHours --attempts 3 --wait 60 2>&1 | Out-String
$code = $LASTEXITCODE
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Encoding utf8 $Log "[$stamp] exit=$code $($output.Trim() -replace '\s*\r?\n\s*', ' | ')"

if ($code -ne 0) {
    $reason = ($output -split "\r?\n" | Where-Object { $_ -match "::error::" } | Select-Object -Last 1) -replace '^.*?failed for \S+: ', ''
    if (-not $reason) { $reason = "check failed (exit $code); see $Log" }
    Show-Alert "Air-quality forecast page needs attention" $reason
}
exit $code
