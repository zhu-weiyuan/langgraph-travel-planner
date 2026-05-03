# -*- coding: utf-8 -*-
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import urllib.request, json

# Test health endpoint
resp = urllib.request.urlopen('http://localhost:7861/api/health', timeout=5)
print("Health:", json.loads(resp.read()))

# Test homepage loads
resp2 = urllib.request.urlopen('http://localhost:7861/', timeout=5)
html = resp2.read().decode('utf-8')
print(f"Homepage: {len(html)} chars, has title: {'旅游规划' in html}")
