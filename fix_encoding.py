# -*- coding: utf-8 -*-
import sys
p = r'C:\Users\Administrator\.openclaw\workspace\langgraph-travel-planner\agent\nodes.py'
with open(p, 'rb') as f:
    data = f.read()
# Replace the problematic line
old = b"""        'Xi'an', 'Chongqing'"""
new = b"""        "Xi'an", "Chongqing" """
data = data.replace(
    b"'Xi'an'",
    b'"Xi\\u0027an"'
)
# Actually simpler: just replace Xi'an with Xian
data = data.replace(b"Xi'an", b"Xian")
with open(p, 'wb') as f:
    f.write(data)
print('Fixed!')
