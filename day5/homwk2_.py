#!/usr/bin/python3
# 作者:xiangyang
# 2025年06月01日00时26分10秒
# zxy3278@163.com

print(f"{'-' * 9}第2题{'-' * 9}")
# 2、求两个有序数字列表的公共元素
list_1 = [1, 2, 3, 4, ]
list_2 = [2, 5, 4, 6, 7, 6]
print(set(list_2) & set(list_1))

print(f"{'-' * 9}第3题{'-' * 9}")
# 3、给定一个n个整型元素的列表a，其中有一个元素出现次数超过n / 2，求这个元素
list_num = [3, 7, 4, 3, 4, 5, 4, 4, ]
list_dedup = list(set(list_num))  # 列表去重
n = 0  # 记录元素出现的最大次数
num = -1  # 记录集合中元素出现最多次数的位置
for i in range(len(list_dedup)):
    if list_num.count(list_dedup[i]) > n:
        n = list_num.count(list_dedup[i])
        num = i
print(f"{list_dedup[num]}这个元素出现次数超过n / 2")

# 4、列表、元组，字典的相同点，不同点有哪些，请罗列
"""
可变性：列表和字典是可变性的；元组是不可变的
存储方式：列表和元组是有序的通过索引访问，字典是无序的
元素类型：列表和元组可以存储任意类型；字典的键是不可变类型，值是可变的
操作：列表和字典都可以增删查改；元组不能改
"""
print(f"{'-' * 9}第5题{'-' * 9}")
# 5、将元组 (1,2,3) 和集合 {4,5,6} 合并成一个列表。
tuple_a = (1, 2, 3)
set_b = {4, 5, 6}
print(list(tuple_a) + list(set_b))

print(f"{'-' * 9}第6题{'-' * 9}")
# 6、在列表 [1,2,3,4,5,6] 首尾分别添加整型元素 7 和 0。
list_6 = [1, 2, 3, 4, 5, 6]
list_6.insert(0, 7)
list_6.append(0)
print(list_6)

print(f"{'-' * 9}第7题{'-' * 9}")
# 7、反转列表 [0,1,2,3,4,5,6,7]
list_7 = [0, 1, 2, 3, 4, 5, 6, 7]
list_7.reverse()
print(list_7)

print(f"{'-' * 9}第8题{'-' * 9}")
# 8、反转列表 [0,1,2,3,4,5,6,7] 后给出中元素 5 的索引号。
list_8 = [0, 1, 2, 3, 4, 5, 6, 7]
list_8.reverse()
print(f"元素 5 的索引号是{list_8.index(5)}")

print(f"{'-' * 9}第8题{'-' * 9}")
# 9、分别统计列表 [True,False,0,1,2] 中 True,False,0,1,2的元素个数，发现了什么？
"""
出现的个数分别为 2，2，2，1
"""
list_9 = [True, False, 0, 1, 2]
print(list_9.count(True), list_9.count(False), list_9.count(0), list_9.count(1), list_9.count(2))

print(f"{'-' * 9}第10题{'-' * 9}")
# 10、从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除元素‘x’。
list_10 = [True, 1, 0, 'x', None, 'x', False, 2, True]
for i in range(list_10.count("x")):
    list_10.remove("x")
print(list_10)

print(f"{'-' * 9}第11题{'-' * 9}")
# 11、从列表 [True,1,0,‘x’,None,‘x’,False,2,True] 中删除索引号为4的元素。
list_11 = [True, 1, 0, 'x', None, 'x', False, 2, True]
list_11.pop(4)
print(list_11)

print(f"{'-' * 9}第12题{'-' * 9}")
# 12、删除列表中索引号为奇数（或偶数）的元素。
list_12 = [1, 2, 3, 4, 5, 6, 7]
special_num12 = '&'  # 对要删除的奇数位置或者偶数位置进行特殊标记
for i in range(len(list_12)):
    if i % 2 == 1:
        list_12[i] = special_num12
list12_dedup = list(set(list_12))  # 列表降重
list12_dedup.remove(special_num12)
print(list12_dedup)

print(f"{'-' * 9}第13题{'-' * 9}")
# 13、清空列表中的所有元素
list_13 = [1, 2, 3, 4, 5, 6, 7]
list_13.clear()
print(list_13)

print(f"{'-' * 9}第14题{'-' * 9}")
# 14、对列表 [3,0,8,5,7] 分别做升序和降序排列。
list_14 = [3, 0, 8, 5, 7]
list_14.sort()  # 升序
print(list_14)
list_14.sort(reverse=True)  # 降序
print(list_14)

print(f"{'-' * 9}第15题{'-' * 9}")
# 15、将列表 [3,0,8,5,7] 中大于 5 元素置为1，其余元素置为0。
list_15 = [3, 0, 8, 5, 7]
i_15 = 0  # 用于访问列表记录下标
for elem in list_15:
    if elem > 5:
        list_15[i_15] = 1
        i_15 += 1
    else:
        list_15[i_15] = 0
        i_15 += 1
print(list_15)

print(f"{'-' * 9}第16题{'-' * 9}")
# 16、遍历列表 [‘x’,‘y’,‘z’]，打印每一个元素及其对应的索引号。
list_16 = ['x', 'y', 'z']
i_16 = 0
for i in list_16:
    print(f"{i}的索引号为{i_16}")
    i_16 += 1

print(f"{'-' * 9}第17题{'-' * 9}")
# 17、将列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 拆分为奇数组和偶数组两个列表。
list_17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list12_1 = list_17[::2]
list12_2 = list_17[1::2]
print(list12_1)
print(list12_2)

print(f"{'-' * 9}第18题{'-' * 9}")
# 18、分别根据每一行的首元素和尾元素大小对二维列表 [[6, 5], [3, 7], [2, 8]] 排序。相当于
list18 = [[6, 5], [3, 7], [2, 8], ]
list18.sort(reverse=True)
print(list18)

print(f"{'-' * 9}第19题{'-' * 9}")
# 19、从列表 [1,4,7,2,5,8] 索引为3的位置开始，依次插入列表 [‘x’,‘y’,‘z’] 的所有元素。
list19 = [1, 4, 7, 2, 5, 8]
list19[3:3] = ['x', 'y', 'z']
print(list19)

print(f"{'-' * 9}第20题{'-' * 9}")
# 20、快速生成由 [5,50) 区间内的整数组成的列表。
list20 = list(range(5, 50))
print(list20)

print(f"{'-' * 9}第21题{'-' * 9}")
# 21、若 a = [1,2,3]，令 b = a，执行 b[0] = 9， a[0]亦被改变。为何？如何避免？----讲了深COPY和浅COPY再做


print(f"{'-' * 9}第22题{'-' * 9}")
# 22、将列表 [‘x’,‘y’,‘z’] 和 [1,2,3] 转成 [(‘x’,1),(‘y’,2),(‘z’,3)] 的形式。
list22_1 = ['x', 'y', 'z']
list22_2 = [1, 2, 3]
list22_result = list(zip(list22_1, list22_2))
print(list22_result)

print(f"{'-' * 9}第23题{'-' * 9}")


# 23、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的键。
def return_dickey_inlist23(dic_def):
    list_23 = list(dic_def.keys())
    return list_23


dic_23 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
print(return_dickey_inlist23(dic_23))

print(f"{'-' * 9}第24题{'-' * 9}")


# 24、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有的值。
def return_dicvalue_inlist24(dic_def):
    list_24 = list(dic_def.values())
    return list_24


dic_24 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
print(return_dicvalue_inlist24(dic_24))

print(f"{'-' * 9}第25题{'-' * 9}")


# 25、以列表形式返回字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中所有键值对组成的元组。
def return_Keyvaluepair_inlist25(dic_def):
    list25 = list(dic_def.items())
    return list25


dic_25 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
print(return_Keyvaluepair_inlist25(dic_25))

print(f"{'-' * 9}第26题{'-' * 9}")
# 26、向字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中追加 ‘David’:19 键值对，更新Cecil的值为17。
dic_26 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
dic_26['David'] = 19
dic_26['Cecil'] = 17
print(dic_26)

print(f"{'-' * 9}第27题{'-' * 9}")
# 27、删除字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中的Beth键后，清空该字典。
dic_27 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
dic_27.pop('Beth')
print(dic_27)
dic_27.clear()
print(dic_27)

print(f"{'-' * 9}第28题{'-' * 9}")
# 28、判断 David 和 Alice 是否在字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21} 中。
dic_28 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
if "David" in dic_28:
    print(f"David在字典中")
else:
    print(f"David不在字典中")
if "Alice" in dic_28:
    print(f"Alice在字典中")
else:
    print(f"Alice不在字典中")

print(f"{'-' * 9}第29题{'-' * 9}")
# 29、遍历字典 {‘Alice’: 20, ‘Beth’: 18, ‘Cecil’: 21}，打印键值对。
dic_29 = {'Alice': 20, 'Beth': 18, 'Cecil': 21}
for key, value in dic_29.items():
    print(f"Key: {key}, Value: {value}")

print(f"{'-' * 9}第31题{'-' * 9}")
# 31、以列表 [‘A’,‘B’,‘C’,‘D’,‘E’,‘F’,‘G’,‘H’] 中的每一个元素为键，默认值都是0，创建一个字典。
list31 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
dict23 = {key: 0 for key in list31}
print(dict23)

print(f"{'-' * 9}第32题{'-' * 9}")
# 32、将二维结构 [[‘a’,1],[‘b’,2]] 和 ((‘x’,3),(‘y’,4)) 转成字典。
list32 = [['a', 1], ['b', 2]]
tuple32 = (('x', 3), ('y', 4))
dict_list32 = dict(list32)
print(dict_list32)
dict_tuple32 = dict(tuple32)
print(dict_tuple32)

print(f"{'-' * 9}第33题{'-' * 9}")
# 33、将元组 (1,2) 和 (3,4) 合并成一个元组。
tuple33_1 = (1, 2)
tuple33_2 = (3, 4)
print(tuple33_1 + tuple33_2)

print(f"{'-' * 9}第34题{'-' * 9}")
# 34、将空间坐标元组 (1,2,3) 的三个元素解包对应到变量 x,y,z。
tuple34 = (1, 2, 3)
x, y, z = tuple34
print(x, y, z)

print(f"{'-' * 9}第35题{'-' * 9}")
# 35、返回元组 (‘Alice’,‘Beth’,‘Cecil’) 中 ‘Cecil’ 元素的索引号。
tuple35 = ('Alice', 'Beth', 'Cecil')
print(tuple35.index("Cecil"))

print(f"{'-' * 9}第36题{'-' * 9}")
# 36、返回元组 (2,5,3,2,4) 中元素 2 的个数。
tuple36 = (2, 5, 3, 2, 4)
print(tuple36.count(2))

print(f"{'-' * 9}第37题{'-' * 9}")
# 37、判断 ‘Cecil’ 是否在元组 (‘Alice’,‘Beth’,‘Cecil’) 中。
tuple37 = ('Alice', 'Beth', 'Cecil')
if "Cecil" in tuple37:
    print(True)
else:
    print(False)

print(f"{'-' * 9}第38题{'-' * 9}")


# 38、返回在元组 (2,5,3,7) 索引号为2的位置插入元素 9 之后的新元组。
def tuple_insert(tuple_38, index, elem):
    """
    元组插入新数据
    :param tuple_38: 需要插入的元组
    :param index: 插入位置
    :param elem: 插入元素
    :return: 元组
    """
    list38 = list(tuple_38)
    list38.insert(index, elem)
    return tuple(list38)


tuple38 = (2, 5, 3, 7)
tuple_inset = tuple_insert(tuple38, 2, 8)
print(tuple_inset)

print(f"{'-' * 9}第39题{'-' * 9}")
# 39、创建一个空集合，增加 {‘x’,‘y’,‘z’} 三个元素。
new_set39 = set()
new_set39.update({'x', 'y', 'z'})
print(new_set39)

print(f"{'-' * 9}第40题{'-' * 9}")
# 40、删除集合 {‘x’,‘y’,‘z’} 中的 ‘z’ 元素，增j加元素 ‘w’，然后清空整个集合。
set40 = {'x', 'y', 'z'}
set40.remove("z")
print(set40)
set40.update("w")
print(set40)
set40.clear()
print(set40)

print(f"{'-' * 9}第41题{'-' * 9}")
# 41、返回集合 {‘A’,‘D’,‘B’} 中未出现在集合 {‘D’,‘E’,‘C’} 中的元素（差集）。
set41_1 = {'A', 'D', 'B'}
set41_2 = {'D', 'E', 'C'}
set41_dif = set41_1.symmetric_difference(set41_2)
print(set41_dif)

print(f"{'-' * 9}第42题{'-' * 9}")
# 42、返回两个集合 {‘A’,‘D’,‘B’} 和 {‘D’,‘E’,‘C’} 的并集。
set42_1 = {'A', 'D', 'B'}
set42_2 = {'D', 'E', 'C'}
set42_dif = set42_1.union(set42_2)
print(set42_dif)

print(f"{'-' * 9}第43题{'-' * 9}")
# 43、返回两个集合 {‘A’,‘D’,‘B’} 和 {‘D’,‘E’,‘C’} 的交集。
set43_1 = {'A', 'D', 'B'}
set43_2 = {'D', 'E', 'C'}
set43_dif = set43_1.intersection(set43_2)
print(set43_dif)

print(f"{'-' * 9}第44题{'-' * 9}")
# 44、返回两个集合 {‘A’,‘D’,‘B’} 和 {‘D’,‘E’,‘C’} 未重复的元素的集合。
set44_1 = {'A', 'D', 'B'}
set44_2 = {'D', 'E', 'C'}
set44_dif = set44_1.symmetric_difference(set44_2)
print(set44_dif)

print(f"{'-' * 9}第45题{'-' * 9}")
# 45、判断两个集合 {‘A’,‘D’,‘B’} 和 {‘D’,‘E’,‘C’} 是否有重复元素。
set45_1 = {'A', 'D', 'B'}
set45_2 = {'D', 'E', 'C'}
if set45_1.isdisjoint(set45_2):
    print(False)
else:
    print(True)

print(f"{'-' * 9}第46题{'-' * 9}")
# 46、判断集合 {‘A’,‘C’} 是否是集合 {‘D’,‘C’,‘E’,‘A’} 的子集
set46_1 = {'A', 'C'}
set46_2 = {'D', 'E', 'C', 'A'}
if set46_1.issubset(set46_2):
    print(True)
else:
    print(False)

print(f"{'-' * 9}第47题{'-' * 9}")
# 47、去除数组 [1,2,5,2,3,4,5,‘x’,4,‘x’] 中的重复元素。
list49 = [1, 2, 5, 2, 3, 4, 5, 'x', 4, 'x']
list49_dedup = list(set(list49))
print(list49_dedup)

print(f"{'-' * 9}第48题{'-' * 9}")
# 48、返回字符串 ‘abCdEfg’ 的全部大写、全部小写和大下写互换形式。
str48 = "abCdEfg"
print(str48.swapcase())
print(str48.lower())
print(str48.upper())

print(f"{'-' * 9}第49题{'-' * 9}")
# 49、判断字符串 ‘abCdEfg’ 是否首字母大写，字母是否全部小写，字母是否全部大写。
str49 = "abCdEfg"
print(f"首字母是否大写 :{str49.istitle()}")
print(f"是否全部小写 :{str49.islower()}")
print(f"是否全部大写 :{str49.isupper()}")

print(f"{'-' * 9}第50题{'-' * 9}")
# 50、返回字符串 ‘this is python’ 首字母大写以及字符串内每个单词首字母大写形式。
str50 = "this is python"
print(str50.title())

print(f"{'-' * 9}第51题{'-' * 9}")
# 51、判断字符串 ‘this is python’ 是否以 ‘this’ 开头，又是否以 ‘python’ 结尾。
str51 = "this is python"
print(f"是否以 ‘this’ 开头 :{str51.startswith("this")}")
print(f"是否以 ‘python’ 结尾 :{str51.endswith("python")}")

print(f"{'-' * 9}第52题{'-' * 9}")
# 52、返回字符串 ‘this is python’ 中 ‘is’ 的出现次数。
print("this is python".count("is"))

print(f"{'-' * 9}第53题{'-' * 9}")
# 53、返回字符串 ‘this is python’ 中 ‘is’ 首次出现和最后一次出现的位置。
print("this is python".find("is"))
print("this is python".rfind("is"))

print(f"{'-' * 9}第54题{'-' * 9}")
# 54、将字符串 ‘this is python’ 切片成3个单词。
print("this is python".split())

print(f"{'-' * 9}第55题{'-' * 9}")
# 55、返回字符串 ‘blog.csdn.net/xufive/article/details/102946961’ 按路径分隔符切片的结果。
print('blog.csdn.net/xufive/article/details/102946961'.split("/"))

print(f"{'-' * 9}第56题{'-' * 9}")
# 56、将字符串 ‘2.72, 5, 7, 3.14’ 以半角逗号切片后，再将各个元素转成浮点型或整形。
list56 = '2.72, 5, 7, 3.14'.split(",")
i = 0
while i < len(list56):  # 去除空格
    if list56[i].startswith(" "):
        list56[i] = list56[i][1:]
        continue
    i += 1
i = 0
while i < len(list56):
    list56[i] = float(list56[i])
    i += 1
print(list56)

print(f"{'-' * 9}第57题{'-' * 9}")
# 57、判断字符串 ‘adS12K56’ 是否完全为字母数字，是否全为数字，是否全为字母？
print(f"是否完全为字母数字 ：{'adS12K56'.isalnum()}")
print(f"是否完全为数字 ：{'adS12K56'.isdecimal()}")
print(f"是否完全为字母 ：{'adS12K56'.isalpha()}")


print(f"{'-' * 9}第58题{'-' * 9}")
# 58、将字符串 ‘there is python’ 中的 ‘is’ 替换为 ‘are’。
str58="there is python".replace("is","are")
print(str58)


print(f"{'-' * 9}第59题{'-' * 9}")
# 59、清除字符串 ‘\t python \n’ 左侧、右侧，以及左右两侧的空白字符。
str59="\t python \n".lstrip("\t ")
str59=str59.rstrip(("\n "))
# str59=str59.lstrip((" "))
# str59=str59.rstrip((" "))
print(str59)



print(f"{'-' * 9}第60题{'-' * 9}")
# 60、将三个全英文字符串（比如，‘ok’, ‘hello’, ‘thank you’）分行打印，实现左对齐、右对齐和居中对齐效果。
str60_1="ok"
str60_2="hello"
str60_3="thank you"
print(len(str60_3))
print(str60_1.ljust(len(str60_3)))
print(str60_2.ljust(len(str60_3)))
print(str60_3.ljust(len(str60_3)))

print(str60_1.rjust(len(str60_3)))
print(str60_2.rjust(len(str60_3)))
print(str60_3.rjust(len(str60_3)))

print(str60_1.center(len(str60_3)))
print(str60_2.center(len(str60_3)))
print(str60_3.center(len(str60_3)))

print(f"{'-' * 9}第61题{'-' * 9}")
# 61、将三个字符串 ‘15’, ‘127’, ‘65535’ 左侧补0成同样长度。



print(f"{'-' * 9}第49题{'-' * 9}")
# 62、将列表 [‘a’,‘b’,‘c’] 中各个元素用’|'连接成一个字符串。
print(f"{'-' * 9}第49题{'-' * 9}")
# 63、将字符串 ‘abc’ 相邻的两个字母之间加上半角逗号，生成新的字符串。
print(f"{'-' * 9}第49题{'-' * 9}")
# 64、从键盘输入手机号码，输出形如 ‘Mobile: 186 6677 7788’ 的字符串。
print(f"{'-' * 9}第49题{'-' * 9}")
# 65、从键盘输入年月日时分秒，输出形如 ‘2019-05-01 12:00:00’ 的字符串。
print(f"{'-' * 9}第49题{'-' * 9}")
# 66、给定两个浮点数 3.1415926 和 2.7182818，格式化输出字符串 ‘pi = 3.1416, e = 2.7183’。
print(f"{'-' * 9}第49题{'-' * 9}")
# 67、将 0.00774592 和 356800000 格式化输出为科学计数法字符串。
print(f"{'-' * 9}第49题{'-' * 9}")
# 68、将列表 [0,1,2,3.14,‘x’,None,’’,list(),{5}] 中各个元素转为布尔型。
print(f"{'-' * 9}第49题{'-' * 9}")
# 69、返回字符 ‘a’ 和 ‘A’ 的ASCII编码值。
print(f"{'-' * 9}第49题{'-' * 9}")
# 70、返回ASCII编码值为 57 和 122 的字符。
print(f"{'-' * 9}第49题{'-' * 9}")
# 71、将列表 [3,‘a’,5.2,4,{},9,[]] 中 大于3的整数或浮点数置为1，其余置为0。
print(f"{'-' * 9}第49题{'-' * 9}")
# 72、将二维列表 [[1], [‘a’,‘b’], [2.3, 4.5, 6.7]] 转为 一维列表。
print(f"{'-' * 9}第49题{'-' * 9}")
# 73、将等长的键列表和值列表转为字典。
print(f"{'-' * 9}第49题{'-' * 9}")
# # 74、数字列表求和。
