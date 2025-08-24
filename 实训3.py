# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:45:20 2025

@author: I is God
"""

import re

# 示例文本
sanguo_text = "《三国志》（卷一）魏书一·武帝纪123 曹操（155年－220年3月15日），字孟德，小名阿瞒...!"

# 过滤特殊符号、标点符号、英文、数字
cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', sanguo_text)

print("过滤后的文本:", cleaned_text)


import re

text = "GuangZhou：510000ShenZhen：518000FoShan：528000ZhuHai：519000；DongWan：523000"

# 提取中文地名
cities = re.findall(r'[\u4e00-\u9fa5]+', text)

# 提取邮编
postcodes = re.findall(r'\d{6}', text)

# 组合地名和邮编
result = dict(zip(cities, postcodes))

print("提取结果:", result)


import re

html_tag = '<meta name="description" content="京东JD.COM﹣专业的综合网上购物商城，销售家电、数码通信、电脑、家居百货、服装服饰、母婴、图书、食品等领域数万个品牌优质商品。便捷、诚信的服务为您提供愉悦的网上购物体验！"/>'

# 提取content属性中的文本
text = re.findall(r'content="([^"]+)"', html_tag)[0]

print("提取的文本:", text)


import re

# (2) 提取"(888)"
phone_text = "(888)555-1234"
area_code = re.findall(r'\(\d+\)', phone_text)[0]
print("提取的区号:", area_code)

# (3) 提取所有电子邮箱地址
email_text = "111111@qq.comabcdefg@126.comabc123@163.com"
emails = re.findall(r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z0-9]+', email_text)
print("提取的邮箱:", emails)