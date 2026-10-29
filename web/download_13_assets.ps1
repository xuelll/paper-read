$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$src = "E:\xuejing\project\paper_read\web\en_src\13_VoT.html"
$assetDir = "E:\xuejing\project\paper_read\web\en_src\13_VoT_assets"
New-Item -ItemType Directory -Force -Path $assetDir | Out-Null

$raw = [System.IO.File]::ReadAllText($src, [System.Text.Encoding]::UTF8)
$s = $raw.IndexOf("<article")
$e = $raw.LastIndexOf("</article>")
$article = $raw.Substring($s, $e - $s + 10)
# 匹配 src="/html/2609.07815/assets/x.png" 与 src="assets/x.png" 两种形式
$srcs = [regex]::Matches($article, 'src="(?:/html/2609\.07815/)?(assets/([^"]+))"') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
Write-Output ("found " + $srcs.Count + " asset srcs")

$okList = @()
$failList = @()
foreach ($rel in $srcs) {
  $sub = $rel -replace "^assets/", ""
  $localPath = Join-Path $assetDir $sub
  $localDir = Split-Path $localPath -Parent
  New-Item -ItemType Directory -Force -Path $localDir | Out-Null
  if ((Test-Path $localPath) -and ((Get-Item $localPath).Length -gt 200)) {
    $okList += $rel
    Write-Output ("CACHED " + $rel)
    continue
  }
  $url = "https://ar5iv.labs.arxiv.org/html/2609.07815/" + $rel
  $done = $false
  for ($i = 1; $i -le 6; $i++) {
    curl.exe -sS -L --max-time 120 --retry 2 --retry-all-errors -o $localPath $url 2>$null
    if ((Test-Path $localPath) -and ((Get-Item $localPath).Length -gt 200)) {
      $done = $true
      break
    }
    Start-Sleep -Seconds 5
  }
  if ($done) {
    $okList += $rel
    Write-Output ("OK " + $rel + " (" + (Get-Item $localPath).Length + " bytes)")
  } else {
    if (Test-Path $localPath) { Remove-Item $localPath -Force -ErrorAction SilentlyContinue }
    $failList += $rel
    Write-Output ("FAIL " + $rel)
  }
}
Write-Output ("=== SUMMARY: " + $okList.Count + " ok / " + $failList.Count + " failed ===")
