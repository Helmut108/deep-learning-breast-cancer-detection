import urllib.request

url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
request = urllib.request.Request(url, method="HEAD")

with urllib.request.urlopen(request, timeout=15) as response:
    print(response.status)  # 200 means success