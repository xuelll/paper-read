$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$papers = @(
  @{n="19_ImageReward"; id="2304.05977"},
  @{n="20_HPS"; id="2303.14420"},
  @{n="21_HPSv2"; id="2306.09341"},
  @{n="22_PickScore"; id="2305.01569"},
  @{n="23_RealGen"; id="2512.00473"},
  @{n="24_TextPecker"; id="2602.20903"},
  @{n="25_UnifiedReward"; id="2503.05236"},
  @{n="26_SigLIP"; id="2303.15343"},
  @{n="27_SpectraReward"; id="2607.11886"}
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
Write-Output "=== HTML DOWNLOAD 19-27 DONE ==="
Get-ChildItem $dir -Filter "*_*.html" | Where-Object { $_.Name -match "^(19|2[0-7])_" } | ForEach-Object { Write-Output ($_.Name + " : " + $_.Length) }
