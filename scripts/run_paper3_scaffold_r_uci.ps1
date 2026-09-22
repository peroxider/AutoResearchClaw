$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$settings = Get-Content 'C:\Users\25824\.claude\settings.json.minimax' -Raw | ConvertFrom-Json
$env:MINIMAX_API_KEY = [string]$settings.env.ANTHROPIC_AUTH_TOKEN

& (Join-Path $projectRoot '.venv\Scripts\python.exe') `
    (Join-Path $projectRoot 'scripts\paper3_scaffold_r_uci_experiment.py') `
    --n 600 --batch-size 30

exit $LASTEXITCODE
