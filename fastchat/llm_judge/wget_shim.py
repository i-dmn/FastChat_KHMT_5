import sys, os, urllib.request
args = sys.argv[1:]
out = args[args.index("-O") + 1]
url = args[-1]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
urllib.request.urlretrieve(url, out)
print("downloaded", out)
