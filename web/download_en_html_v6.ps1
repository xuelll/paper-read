$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
$papers = @(
  @{n="43_VisTA"; id="2505.20289"},
  @{n="44_Vstar-RFT"; id="2505.14246"},
  @{n="45_Thyme"; id="2508.11630"},
  @{n="46_PixelReasoner"; id="2505.15966"},
  @{n="47_RVPT"; id="2509.01656"},
  @{n="48_SFT-vs-RL"; id="2501.17161"},
  @{n="49_ToRL"; id="2503.23383"},
  @{n="50_DeepEyes"; id="2505.14362"},
  @{n="51_Vstar"; id="2312.14135"},
  @{n="52_PyVision"; id="2507.07998"}
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
  Start-Sleep -Seconds 2
}
Write-Output "=== HTML DOWNLOAD 43-52 DONE ==="
