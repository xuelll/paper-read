$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dir = "E:\xuejing\project\paper_read\web\en_src"
$out = Join-Path $dir "42_CodeV.html"
$urls = @(
  "https://arxiv.org/html/2511.19661",
  "https://arxiv.org/html/2511.19661v1",
  "https://ar5iv.labs.arxiv.org/html/2511.19661"
)
$ok = $false
foreach ($u in $urls) {
  if ($ok) { break }
  curl.exe -sS -L --max-time 300 --retry 3 --retry-all-errors -o $out $u 2>$null
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 50000)) {
    $c = [System.IO.File]::ReadAllText($out)
    if ($c -match "ltx_document") { $ok = $true; Write-Output ("42_CodeV OK " + (Get-Item $out).Length + " bytes <- " + $u) }
  }
  if (-not $ok -and (Test-Path $out)) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
}
if (-not $ok) { Write-Output "42_CodeV FAILED (all sources)" }
