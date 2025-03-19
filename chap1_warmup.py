"""
1.导入numpy库
"""
import numpy as np
import matplotlib.pyplot as plt

"""
2.建立一个一维数组 a 初始化为[4,5,6], 
(1)输出a 的类型（type）
(2)输出a的各维度的大小（shape）
(3)输出 a的第一个元素（值为4）
"""
a2 = np.array([4, 5, 6])
print(type(a2))       
print(a2.shape)           
print(a2[0])         

"""
3.建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]]
(1)输出各维度的大小（shape）
(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
"""
b3 = np.array([[4, 5, 6], [1, 2, 3]])
print(b3.shape)          
print(b3[0, 0], b3[0, 1], b3[1, 1]) 

"""
4. (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）
(2)建立一个全1矩阵b,大小为4x5;
 (3)建立一个单位矩阵c ,大小为4x4; 
(4)生成一个随机数矩阵d,大小为 3x2.
"""
a4 = np.zeros((3, 3), dtype=int)
b4 = np.ones((4, 5))
c4 = np.eye(4)
d4 = np.random.random((3, 2))


"""
5. 建立一个数组 a，初始化为 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
"""
a5= np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

"""
6. 把上一题的 a 数组的 0到1行，2到3列，放到 b 里面去
"""
b6 = a5[0:2, 2:4]

"""
7. 把第5题中数组 a 的最后两行所有元素放到 c 中
"""
c7 = a5[1:, :]

"""
8. 建立数组 a，初始化为 [[1, 2], [3, 4], [5, 6]]，
输出 (0,0) (1,1) (2,0) 这三个元素
"""
a8 = np.array([[1, 2], [3, 4], [5, 6]])
print(a8[[0, 1, 2], [0, 1, 0]])

"""
9. 建立矩阵 a，初始化为 [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，
输出 (0,0), (1,2), (2,0), (3,1)
"""
a9 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a9[[0, 1, 2, 3], [0, 2, 0, 1]])

"""
10. 对9中输出的那四个元素，每个都加上10，然后重新输出矩阵 a
"""
a9[[0, 1, 2, 3], [0, 2, 0, 1]] += 10
print(a9)

"""
11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型
"""
x11 = np.array([1, 2])
print(x11.dtype)

"""
12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型
"""
x12 = np.array([1.0, 2.0])
print(x12.dtype)


"""
13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]],
 dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
"""
x13 = np.array([[1, 2], [3, 4]], dtype=np.float64)
y13 = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x13 + y13)
print(np.add(x13, y13))


"""
14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
"""
print(x13- y13)
print(np.subtract(x13, y13))

"""
15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 
还有 np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
"""
print(x13 * y13) #形状相同的方阵 按元素相乘
print(np.multiply(x13, y13)) #形状相同的方阵 按元素相乘
print(np.dot(x13, y13))  #形状可以不一样 矩阵乘法

"""
16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())
"""
print(np.divide(x13, y13))  

"""
17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )
"""
print(np.sqrt(x13)) 

"""
18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
"""
print(x13.dot(y13))  
print(np.dot(x13, y13))  

"""
19.利用13题目中的 x,进行求和。提示：输出三种求和 
(1)print(np.sum(x)): (2)print(np.sum(x，axis =0 )); (3)print(np.sum(x,axis = 1))
"""
print(np.sum(x13))  # 对整个数组求和
print(np.sum(x13, axis=0))  # 对每一列求和
print(np.sum(x13, axis=1))  # 对每一行求和

"""
20.利用13题目中的 x,进行求平均数（提示：输出三种平均数
(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
"""
print(np.mean(x13))  # 对整个数组求平均数
print(np.mean(x13, axis=0))  # 对每一列求平均数
print(np.mean(x13, axis=1))  # 对每一行求平均数

"""
21.利用13题目中的x，对x 进行矩阵转置，
然后输出转置后的结果，（提示： x.T 表示对 x 的转置）
"""
print(x13.T) 


"""
22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）
"""
print(np.exp(x13))  

"""
23.利用13题目中的 x,求值最大的下标（提示
(1)print(np.argmax(x)) ,
(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))
"""
print(np.unravel_index(np.argmax(x13), x13.shape))

"""
24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) 
（提示这里用到 matplotlib.pyplot 库）
"""
x24 = np.arange(0, 100, 0.1)
y24 = x24 ** 2
plt.plot(x24, y24)
plt.title('Plot of y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

"""
25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)
(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)
"""
x25 = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x25)
y_cos = np.cos(x25)
plt.plot(x25, y_sin, label='sin(x)', color='blue')
plt.plot(x25, y_cos, label='cos(x)', color='red')
plt.title('Plot of sin(x) and cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
