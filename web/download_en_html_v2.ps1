$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$papers = @(
  @{n="15_EditReward"; id="2509.26346"},
  @{n="16_EditScore"; id="2509.23909"},
  @{n="17_FIRM-Reward"; id="2603.12247"},
  @{n="18_SpatialReward"; id="2602.07458"}
)
foreach ($p in $papers) {
  $out = Join-Path $dir ($p.n + ".html")
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 50000)) {
    $c = [System.IO.File]::ReadAllText($out)
    if ($c -match "ltx_document") { Write-Output ("SKIP " + $p.n); continue }
  }
  $urls = @(
    ("https://arxiv.org/html/" + $p.id),
    ("https://arxiv.org/html/" + $p.id + "v1"),
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
Get-ChildItem $dir -Filter "1[5-8]_*.html" | ForEach-Object { Write-Output ($_.Name + " : " + $_.Length) }
