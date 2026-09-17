param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$SafeVersion,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = "Stop"

$buildRoot = Join-Path $env:RUNNER_TEMP "facetrackvr-windows-build"
$distDir = Join-Path $buildRoot "dist"
$workDir = Join-Path $buildRoot "work"
$stageDir = Join-Path $buildRoot "stage\FaceTrackVR-$SafeVersion"

New-Item -ItemType Directory -Force -Path $distDir, $workDir, $stageDir, $OutDir | Out-Null

Write-Host "[windows-build] installing locked dependencies..."
Push-Location $RepoRoot
try {
    uv sync --locked --all-groups

    Write-Host "[windows-build] running PyInstaller..."
    uv run pyinstaller `
        --noconfirm `
        --clean `
        --distpath $distDir `
        --workpath $workDir `
        EyeTrackApp\eyetrackapp.spec
}
finally {
    Pop-Location
}

$executable = Join-Path $distDir "eyetrackapp.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Bundle missing eyetrackapp.exe"
}

Write-Host "[windows-build] smoke testing bundled executable..."
$process = Start-Process -FilePath $executable -WorkingDirectory $distDir -PassThru -WindowStyle Hidden
if ($process.WaitForExit(15000)) {
    if ($process.ExitCode -ne 0) {
        throw "Bundled executable exited with code $($process.ExitCode)"
    }
}
else {
    Stop-Process -Id $process.Id
    $process.WaitForExit()
}

Copy-Item -LiteralPath $executable -Destination $stageDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $stageDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $stageDir
Set-Content -LiteralPath (Join-Path $stageDir "VERSION") -Value $Version -Encoding ascii

$archive = Join-Path $OutDir "FaceTrackVR-$SafeVersion-windows-x86_64.zip"
Compress-Archive -Path $stageDir -DestinationPath $archive
Write-Host "[windows-build] wrote $archive"
