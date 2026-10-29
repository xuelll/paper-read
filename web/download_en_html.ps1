$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$papers = @(
  @{n="01_RubricRM"; id="2608.26956"; v="v2"},
  @{n="02_VA-Judger"; id="2608.18607"; v="v4"},
  @{n="03_TryOnReward"; id="2609.13259"; v="v1"},
  @{n="04_Lever-Edit"; id="2608.22780"; v="v1"},
  @{n="05_RewardVerse"; id="2609.22947"; v="v2"},
  @{n="06_FIRM-Video"; id="2608.21839"; v="v1"},
  @{n="07_SenseNova-U1.5"; id="2609.11929"; v="v1"},
  @{n="09_VicEdit"; id="2608.16745"; v="v1"},
  @{n="10_UniSpace"; id="2608.08676"; v="v1"},
  @{n="11_CapabilityDataDesign"; id="2608.18076"; v="v2"},
  @{n="12_DensePred2Editing"; id="2608.14740"; v="v2"},
  @{n="13_VoT"; id="2609.07815"; v="v1"},
  @{n="14_TRM"; id="2609.37372"; v="v2"}
)
foreach ($p in $papers) {
  $out = Join-Path $dir ($p.n + ".html")
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 50000)) {
    $c = [System.IO.File]::ReadAllText($out)
    if ($c -match "ltx_document") { Write-Output ("SKIP " + $p.n); continue }
  }
  $urls = @(
    ("https://arxiv.org/html/" + $p.id + $p.v),
    ("https://arxiv.org/html/" + $p.id),
    ("https://ar5iv.labs.arxiv.org/html/" + $p.id)
  )
  $ok = $false
  foreach ($u in $urls) {
    if ($ok) { break }
    curl.exe -sS -L --max-time 300 --retry 3 --retry-all-errors -o $out $u 2>$null
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 50000)) {
      $c = [System.IO.File]::ReadAllText($out)
      if ($c -match "ltx_document") {
        $ok = $true
        Write-Output ($p.n + " OK " + (Get-Item $out).Length + " bytes <- " + $u)
      }
    }
    if (-not $ok -and (Test-Path $out)) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
  }
  if (-not $ok) { Write-Output ($p.n + " FAILED (all sources)") }
  Start-Sleep -Seconds 1
}
Write-Output "=== HTML DOWNLOAD DONE ==="
Get-ChildItem $dir | ForEach-Object { Write-Output ($_.Name + " : " + $_.Length) }
