$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$papers = @(
  @{n="28_CLIP"; id="2103.00020"},
  @{n="29_CLIPScore"; id="2104.08718"},
  @{n="30_HPSv3"; id="2508.03789"},
  @{n="31_VisionReward"; id="2412.21059"},
  @{n="32_UnifiedReward-Think"; id="2505.03318"},
  @{n="33_UnifiedReward-Flex"; id="2602.02380"},
  @{n="34_InstructPix2Pix"; id="2211.09800"},
  @{n="35_AutoRubric"; id="2603.00077"},
  @{n="36_RubricHub"; id="2601.08430"},
  @{n="37_OpenRubrics"; id="2510.07743"},
  @{n="38_AutoRubric-R1V"; id="2510.14738"},
  @{n="39_Proxy-GRM"; id="2603.16600"},
  @{n="40_AutoRubric-T2I"; id="2605.17602"},
  @{n="41_AutoRubric-Reward"; id="2605.08354"}
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
Write-Output "=== HTML DOWNLOAD 28-41 DONE ==="
Get-ChildItem $dir -Filter "*.html" | Where-Object { $_.Name -match "^(2[89]|3[0-9]|4[01])_" } | ForEach-Object { Write-Output ($_.Name + " : " + $_.Length) }
