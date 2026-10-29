# build_web.ps1 - assemble the bilingual reading site
# NOTE: keep this file ASCII-only; all Chinese strings live in papers_meta.json (UTF-8)
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = "E:\xuejing\project\paper_read"
$web = Join-Path $root "web"
$pagesDir = Join-Path $web "papers"
New-Item -ItemType Directory -Force -Path $pagesDir | Out-Null

function Read-Utf8([string]$path) {
    return [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
}
function Write-Utf8([string]$path, [string]$content) {
    [System.IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding($false)))
}

$meta = Read-Utf8 (Join-Path $web "papers_meta.json") | ConvertFrom-Json
$template = Read-Utf8 (Join-Path $web "template_paper.html")

$indexCards = New-Object System.Collections.Generic.List[string]

foreach ($p in $meta.papers) {
    $num = $p.n.Substring(0, 2)
    $pageFile = Join-Path $pagesDir ($p.n + ".html")
    $arxivUrl = "https://arxiv.org/abs/" + $p.id

    # ---------- English side ----------
    $enFile = Join-Path $web ("en_src\" + $p.n + ".html")
    $enHtml = ""
    $enSrcNote = ""
    $enTitle = [string]$p.en
    if ((Test-Path $enFile) -and ((Get-Item $enFile).Length -gt 50000)) {
        $raw = Read-Utf8 $enFile
        if ($raw -match "ltx_document") {
            # base url for relative assets
            $base = ""
            $bm = [regex]::Match($raw, '<base\s+href="([^"]+)"')
            if ($bm.Success) {
                $base = $bm.Groups[1].Value
                if (-not $base.EndsWith("/")) { $base = $base + "/" }
            }
            if ($base -eq "") {
                # infer versioned base from image paths like src="2608.26956v2/x.png"
                $vm = [regex]::Match($raw, '(?:src|href)="(\d{4}\.\d{4,5}v\d+)/')
                if ($vm.Success) { $base = "https://arxiv.org/html/" + $vm.Groups[1].Value + "/" }
            }
            if ($base -eq "") { $base = "https://arxiv.org/html/" + $p.id + "/" }

            # extract <article> ... </article>
            $s = $raw.IndexOf("<article")
            $e = $raw.LastIndexOf("</article>")
            if ($s -ge 0 -and $e -gt $s) {
                $enHtml = $raw.Substring($s, $e - $s + 10)
            } else {
                $s2 = $raw.IndexOf("<body")
                $e2 = $raw.LastIndexOf("</body>")
                if ($s2 -ge 0 -and $e2 -gt $s2) { $enHtml = $raw.Substring($s2, $e2 - $s2) }
            }
            # strip script/style blocks
            $enHtml = [regex]::Replace($enHtml, "(?is)<script.*?</script>", "")
            $enHtml = [regex]::Replace($enHtml, "(?is)<style.*?</style>", "")

            # detect ar5iv-sourced file (assets not hosted on arxiv.org)
            $isAr5iv = $false
            if ($raw -match "ar5iv\.labs\.arxiv\.org") { $isAr5iv = $true }

            if ($isAr5iv) {
                # 1) 把两种形式的资源 src 统一改写为 ar5iv 绝对 URL
                $enHtml = $enHtml -replace 'src="(?:/html/\d{4}\.\d{4,5}(?:v\d+)?/)?(assets/[^"]+)"', ('src="https://ar5iv.labs.arxiv.org/html/' + $p.id + '/$1"')
                # 2) 本地已缓存的资源替换为本地相对路径（离线可用）
                $assetRoot = Join-Path $web ("en_src\" + $p.n + "_assets")
                if (Test-Path $assetRoot) {
                    $localFiles = @()
                    foreach ($lf2 in (Get-ChildItem $assetRoot -Recurse -File)) {
                        if ($lf2.Length -lt 500) { continue }
                        $b2 = $null
                        try {
                            $fs2 = [System.IO.File]::OpenRead($lf2.FullName)
                            $b2 = New-Object byte[] 4
                            $null = $fs2.Read($b2, 0, 4)
                            $fs2.Close()
                        } catch { continue }
                        $isImg2 = ($b2[0] -eq 0x89 -and $b2[1] -eq 0x50) -or ($b2[0] -eq 0xFF -and $b2[1] -eq 0xD8) -or ($b2[0] -eq 0x47 -and $b2[1] -eq 0x49)
                        if ($isImg2) { $localFiles += $lf2 }
                    }
                    foreach ($lf in $localFiles) {
                        $relInAssets = $lf.FullName.Substring($assetRoot.Length + 1).Replace('\', '/')
                        $remote = 'src="https://ar5iv.labs.arxiv.org/html/' + $p.id + '/assets/' + $relInAssets + '"'
                        $localRef = 'src="../en_src/' + $p.n + '_assets/' + $relInAssets + '"'
                        $enHtml = $enHtml.Replace($remote, $localRef)
                    }
                }
                # 3) 其余相对引用 -> ar5iv base（排除已是本地 ../en_src 或绝对的 URL）
                $ar5ivBase = "https://ar5iv.labs.arxiv.org/html/" + $p.id + "/"
                $enHtml = $enHtml -replace 'src="/(?!/)', 'src="https://ar5iv.labs.arxiv.org/'
                $enHtml = $enHtml -replace 'src="(?!https?://|data:|#|mailto:|\.\./)', ('src="' + $ar5ivBase)
            } else {
                # arxiv.org 源文件：
                # 1) src="2608.26956v2/x.png" (paper-id prefixed) -> https://arxiv.org/html/2608.26956v2/x.png
                $enHtml = $enHtml -replace 'src="(\d{4}\.\d{4,5}v\d+)/', 'src="https://arxiv.org/html/$1/'
                # 2) root-relative -> https://arxiv.org/...
                $enHtml = $enHtml -replace 'src="/(?!/)', 'src="https://arxiv.org/'
                $enHtml = $enHtml -replace 'href="/html/', 'href="https://arxiv.org/html/'
                # 3) other relative -> base
                $enHtml = $enHtml -replace 'src="(?!https?://|data:|#|mailto:)', ('src="' + $base)
            }
            # extract the paper title from the first h1 in the article
            $tm = [regex]::Match($enHtml, "(?is)<h1[^>]*>(.*?)</h1>")
            if ($tm.Success) {
                $t = $tm.Groups[1].Value
                $t = [regex]::Replace($t, "<[^>]+>", "")
                $t = [System.Net.WebUtility]::HtmlDecode($t).Trim()
                if ($t.Length -gt 3) { $enTitle = $t }
            }
            $enSrcNote = "(arXiv HTML)"
        }
    }
    if ($enHtml -eq "") {
        # no HTML available: notice card + arXiv link (local PDFs are not published with the site)
        $enHtml = [string]$meta.noHtmlPdfNote + "`n" +
            '<p style="margin:16px 0 0"><a href="' + $arxivUrl + '" target="_blank" rel="noopener">Read on arXiv (' + $p.id + ') &rarr;</a></p>'
        $enSrcNote = "(arXiv link)"
    }

    # ---------- Chinese side ----------
    $zhFile = Join-Path $root ("translations\" + $p.n + [string]$meta.zhSuffix)
    $zhMd = ""
    $zhDone = $false
    if ((Test-Path $zhFile) -and ((Get-Item $zhFile).Length -gt 2000)) {
        $zhMd = Read-Utf8 $zhFile
        $zhDone = $true
    } else {
        $zhMd = [string]$meta.zhPendingNote
    }
    $zhMd = $zhMd.Replace('</script', '<\/script')

    # ---------- fill template ----------
    $page = $template
    $page = $page.Replace('@@NUM@@', $num)
    $page = $page.Replace('@@TITLE@@', [string]$p.zh)
    $page = $page.Replace('@@ARXIVURL@@', $arxivUrl)
    $page = $page.Replace('@@ENSRC@@', $enSrcNote)
    $page = $page.Replace('@@EN@@', $enHtml)
    $page = $page.Replace('@@ZH@@', $zhMd)
    Write-Utf8 $pageFile $page

    # ---------- index card ----------
    $statusHtml = '<span class="c-status">' + [string]$meta.statusDone + '</span>'
    if (-not $zhDone) { $statusHtml = '<span class="c-status pending">' + [string]$meta.statusPending + '</span>' }
    $tcClass = ""
    if ($p.tc -ne 1) { $tcClass = " t" + $p.tc }
    if ($enTitle.Length -gt 110) { $enTitle = $enTitle.Substring(0, 110) + "..." }
    # derive publication year-month from the arXiv id (YYMM.NNNNN)
    $pubDate = "20" + $p.id.Substring(0, 2) + "-" + $p.id.Substring(2, 2)
    $card = '<a class="card" href="papers/' + $p.n + '.html">' +
        '<span class="c-num">' + $num + '</span>' +
        '<span class="c-tag' + $tcClass + '">' + $p.tag + '</span>' +
        '<h3>' + $num + ' · ' + $p.zh + '</h3>' +
        '<div class="c-date">arXiv ' + $pubDate + '</div>' +
        '<div class="c-en">' + $enTitle + '</div>' +
        '<div class="c-meta"><span>arXiv ' + $p.id + '</span>' + $statusHtml + '</div></a>'
    $indexCards.Add($card)
    $zhState = "pending"
    if ($zhDone) { $zhState = "ok" }
    Write-Output ("built " + $p.n + " (en:" + $enSrcNote + ", zh:" + $zhState + ")")
}

# ---------- summary page ----------
$sumSrc = Join-Path $root ([string]$meta.summaryFile)
if (Test-Path $sumSrc) {
    $sumTpl = Read-Utf8 (Join-Path $web "template_summary.html")
    $sumMd = Read-Utf8 $sumSrc
    $sumMd = $sumMd.Replace('</script', '<\/script')
    $sumPage = $sumTpl.Replace('@@ZH@@', $sumMd)
    Write-Utf8 (Join-Path $web "summary.html") $sumPage
    Write-Output "built summary.html"
}

# ---------- index page ----------
$cardsHtml = ($indexCards.ToArray()) -join "`n"
$indexHtml = @'
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@INDEX_TITLE@@</title>
<link rel="stylesheet" href="assets/style.css?v=6">
</head>
<body>
<div class="home">
  <div class="home-hero">
    <h1>@@INDEX_H1@@</h1>
    <p>@@HERO@@</p>
  </div>
  <div class="cards">
    <a class="card" href="summary.html">
      <span class="c-num">&Sigma;</span>
      <span class="c-tag">@@SUM_TAG@@</span>
      <h3>@@SUM_ZH@@</h3>
      <div class="c-en">@@SUM_EN@@</div>
      <div class="c-meta"><span>@@SUM_META@@</span><span class="c-status">@@SUM_STATUS@@</span></div>
    </a>
@@CARDS@@
  </div>
  <div class="home-foot">@@FOOT@@</div>
</div>
</body>
</html>
'@
$indexHtml = $indexHtml.Replace('@@INDEX_TITLE@@', [string]$meta.indexTitle)
$indexHtml = $indexHtml.Replace('@@INDEX_H1@@', [string]$meta.indexH1)
$indexHtml = $indexHtml.Replace('@@SUM_TAG@@', [string]$meta.summaryCardTag)
$indexHtml = $indexHtml.Replace('@@SUM_ZH@@', [string]$meta.summaryCardZh)
$indexHtml = $indexHtml.Replace('@@SUM_EN@@', [string]$meta.summaryCardEn)
$indexHtml = $indexHtml.Replace('@@SUM_META@@', [string]$meta.summaryCardMeta)
$indexHtml = $indexHtml.Replace('@@SUM_STATUS@@', [string]$meta.summaryCardStatus)
$indexHtml = $indexHtml.Replace('@@HERO@@', [string]$meta.indexHero)
$indexHtml = $indexHtml.Replace('@@CARDS@@', $cardsHtml)
$indexHtml = $indexHtml.Replace('@@FOOT@@', [string]$meta.indexFoot)
Write-Utf8 (Join-Path $web "index.html") $indexHtml
Write-Output "built index.html"
Write-Output "=== BUILD DONE ==="
