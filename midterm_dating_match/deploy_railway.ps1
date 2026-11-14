param(
    [string]$PublicUrl
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$railway = (Get-Command railway -ErrorAction SilentlyContinue).Source
if (-not $railway) {
    $localCli = Join-Path $PSScriptRoot ".railway-cli\node_modules\.bin\railway.cmd"
    if (Test-Path $localCli) { $railway = $localCli }
}
if (-not $railway) {
    throw "Railway CLI is missing. Install it with: npm install -g @railway/cli"
}

# Railway detects the root Dockerfile and builds the image on its servers.
& $railway up -y
if ($LASTEXITCODE -ne 0) { throw "Railway deployment failed." }

# A new service has no public domain until one is generated.
if (-not $PublicUrl) {
    & $railway domain
    if ($LASTEXITCODE -ne 0) { throw "Railway domain creation failed." }
}

if ($PublicUrl) {
    python smoke_test.py --url $PublicUrl --evidence results/railway_smoke.json
    if ($LASTEXITCODE -ne 0) { throw "Public API verification failed." }
} else {
    Write-Host "Copy the generated public URL and verify it with:"
    Write-Host "python smoke_test.py --url https://YOUR-SERVICE.up.railway.app --evidence results/railway_smoke.json"
}
