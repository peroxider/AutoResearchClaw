param(
    [string]$FromStage = '',
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot 'artifacts\paper3_medical_llm_audit\background'
$logPath = Join-Path $logDir 'runner.log'
try {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    "[$(Get-Date -Format o)] Background runner initialised as $([Security.Principal.WindowsIdentity]::GetCurrent().Name)." | Tee-Object -FilePath $logPath -Append
    $settings = Get-Content 'C:\Users\25824\.claude\settings.json.minimax' -Raw | ConvertFrom-Json
    $env:MINIMAX_API_KEY = [string]$settings.env.ANTHROPIC_AUTH_TOKEN
    "[$(Get-Date -Format o)] Starting AutoResearchClaw 23-stage Paper 3 run." | Tee-Object -FilePath $logPath -Append
    # ARC writes normal warnings to stderr.  Do not let PowerShell's `Stop`
    # policy turn those advisories into a terminating NativeCommandError.
    $ErrorActionPreference = 'Continue'
    $cliArgs = @('-m', 'researchclaw.cli', 'run', '--config', (Join-Path $projectRoot 'config.paper3_medical_llm_audit.yaml'), '--auto-approve')
    if ($FromStage) {
        # The CLI accepts Stage enum names while operators commonly use the
        # numbered 23-stage notation.  Normalize both forms here so a resume
        # command cannot fail before it reaches the pipeline.
        $stageNames = @{
            '1'='TOPIC_INIT'; '2'='PROBLEM_DECOMPOSE'; '3'='SEARCH_STRATEGY';
            '4'='LITERATURE_COLLECT'; '5'='LITERATURE_SCREEN'; '6'='KNOWLEDGE_EXTRACT';
            '7'='SYNTHESIS'; '8'='HYPOTHESIS_GEN'; '9'='EXPERIMENT_DESIGN';
            '10'='CODE_GENERATION'; '11'='RESOURCE_PLANNING'; '12'='EXPERIMENT_RUN';
            '13'='ITERATIVE_REFINE'; '14'='RESULT_ANALYSIS'; '15'='RESEARCH_DECISION';
            '16'='PAPER_OUTLINE'; '17'='PAPER_DRAFT'; '18'='PEER_REVIEW';
            '19'='PAPER_REVISION'; '20'='QUALITY_GATE'; '21'='KNOWLEDGE_ARCHIVE';
            '22'='EXPORT_PUBLISH'; '23'='CITATION_VERIFY'
        }
        $normalisedStage = if ($stageNames.ContainsKey($FromStage)) { $stageNames[$FromStage] } else { $FromStage.ToUpperInvariant() }
        $cliArgs += @('--from-stage', $normalisedStage)
    }
    if ($OutputDir) { $cliArgs += @('--output', $OutputDir) }
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') @cliArgs 2>&1 |
        Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "AutoResearchClaw exited with code $exitCode."
    }
    "[$(Get-Date -Format o)] Runner finished with exit code $exitCode." | Tee-Object -FilePath $logPath -Append
    exit $exitCode
} catch {
    New-Item -ItemType Directory -Force -Path $logDir -ErrorAction SilentlyContinue | Out-Null
    "[$(Get-Date -Format o)] Runner failed before pipeline start: $($_.Exception.Message)" | Out-File -FilePath $logPath -Append -Encoding utf8
    exit 1
}
