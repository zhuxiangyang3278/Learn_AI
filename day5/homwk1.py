#!/usr/bin/python3
# 作者:xiangyang
# 2025年05月31日23时19分19秒
# zxy3278@163.com

num_str = "0123456789"
print(num_str[2:6])
print(num_str[2:])
print(num_str[:6])
print(num_str[:])
print(num_str[1::2])
print(num_str[2:-1])
print(num_str[-2:])
print(num_str[9::-1])
print(''.join(reversed(num_str)))

print('-'*10)

a = (1, 2, 3)
b = ('a', 'b', 'c')
print(list(zip(a,b)))

print('-'*10)

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list2=list(enumerate(seasons))
print(list2)




