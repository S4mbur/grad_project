param(
    [switch]$NoBuild,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath)
$projectFile = Join-Path $projectRoot "app_desktop\SkinSight.Desktop.csproj"
$deployDir = Join-Path $env:LOCALAPPDATA "SkinSightDesktop\app"
$exePath = Join-Path $deployDir "SkinSight.exe"

Write-Host "SkinSight Desktop launcher"
Write-Host "Project root: $projectRoot"
Write-Host "Deploy dir:   $deployDir"

if (-not $NoBuild) {
    dotnet publish $projectFile -c Debug -r win-x64 --self-contained false -o $deployDir
}

if (-not (Test-Path $exePath)) {
    throw "SkinSight.exe was not found at $exePath"
}

if ($NoStart) {
    Write-Host "Build/publish complete. Not starting app because -NoStart was supplied."
    exit 0
}

$env:SKINSIGHT_PROJECT_ROOT = $projectRoot
$env:SKINSIGHT_BACKEND_MODE = "python"
$env:SKINSIGHT_WSL_DISTRO = "Ubuntu-22.04"
if ($projectRoot -match '^\\\\wsl\.localhost\\[^\\]+\\(.+)$') {
    $env:SKINSIGHT_WSL_PROJECT_ROOT = "/" + ($Matches[1] -replace "\\", "/")
} else {
    $env:SKINSIGHT_WSL_PROJECT_ROOT = "/home/byalc/phase1_project"
}
$env:SKINSIGHT_PYTHON_ENV = "/home/byalc/phase1_env"
[Environment]::SetEnvironmentVariable("SKINSIGHT_PROJECT_ROOT", $projectRoot, "Process")
[Environment]::SetEnvironmentVariable("SKINSIGHT_BACKEND_MODE", $env:SKINSIGHT_BACKEND_MODE, "Process")
[Environment]::SetEnvironmentVariable("SKINSIGHT_WSL_DISTRO", $env:SKINSIGHT_WSL_DISTRO, "Process")
[Environment]::SetEnvironmentVariable("SKINSIGHT_WSL_PROJECT_ROOT", $env:SKINSIGHT_WSL_PROJECT_ROOT, "Process")
[Environment]::SetEnvironmentVariable("SKINSIGHT_PYTHON_ENV", $env:SKINSIGHT_PYTHON_ENV, "Process")

Write-Host "Starting SkinSight Desktop..."
Write-Host "Backend mode:  $env:SKINSIGHT_BACKEND_MODE"
Write-Host "WSL project:   $env:SKINSIGHT_WSL_PROJECT_ROOT"
Start-Process -FilePath $exePath -WorkingDirectory $deployDir
Write-Host "Started. If no window appears, check: $env:TEMP\skinsight_desktop_startup.log"
