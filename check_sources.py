import httpx, urllib.request

URLS = {
    "ctgov":  "https://clinicaltrials.gov/api/v2/stats/size",
    "epmc":   "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=cancer&format=json&resultType=idlist&pageSize=1",
    "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed&retmode=json",
}
UA = "CEWS/0.1 (local competitive-intelligence proof of concept)"

for name, url in URLS.items():
    try:
        r = httpx.get(url, headers={"User-Agent": UA}, timeout=30)
        print(f"{name:7} httpx  {r.status_code} {r.text[:120]!r}")
    except Exception as e:
        print(f"{name:7} httpx  ERROR {type(e).__name__}: {e}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"{name:7} urllib {resp.status} {resp.read(120)!r}")
    except Exception as e:
        print(f"{name:7} urllib ERROR {type(e).__name__}: {e}")