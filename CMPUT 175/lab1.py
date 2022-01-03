# def exercise1():
#     numbers = [11, 25, 32, 4, 67, 18, 50, 67, 4, 11] 
#     oddNumbers=[]
#     print("The content of object",id(oddNumbers),"are",oddNumbers)
#     [oddNumbers.append(i) for i in numbers if i%2!=0]
#     print("The content of object",id(oddNumbers),"are",oddNumbers)
#     oddNumbers.sort(reverse=True)
#     print("The smallest odd number",oddNumbers.pop(-1),"has been removed from the list of odd numbers.")
#     print("The smallest odd number",oddNumbers.pop(0)," has been removed from the list of odd numbers.")
#     print("The contents of object",id(oddNumbers)," have been updated to",oddNumbers)
#     print("There are ",len(numbers),"numbers in the original list")
#     [numbers.remove(i) for i in numbers if max(oddNumbers)==i]
#     print("After removing the largest odd number, there are ",len(numbers)," numbers in the list:\n",numbers)
# exercise1()

# def exercise2():
#     months = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT') 
#     print("The contents of object",id(months),"are",months)
#     months+=('NOV','DEC')
#     print("The contents of object",id(months),"are",months)
#     precipitation2020 = [21.3, 17.9, 12.9, 2.8, 92.9, 105.5,121.2, 5.0, 11.2, 22.5, 2.5] 
#     precipitation2020.insert(7,83)
#     dictionary=dict(zip(months,precipitation2020))
#     for i in range(len(dictionary)):
#         if months[i]=="MAY":
#             print(precipitation2020[i],"mm fell in",months[i])
#     inp=input("Please enter a month:")
#     for i in range(len(dictionary)):
#         if inp==months[i]:
#             print(precipitation2020[i],"mm fell in",months[i])
# exercise2()

# def exercise3():
#     animals={'dog','cat','fish','snake'}
#     print("Teh content of object",id(animals),"are",animals)
#     animals.discard('snake')
#     animals.add('birds')
#     print("The contents of",id(animals),"are",animals)
#     alice_ani={"dog","cat","rabbit","hamster"}
#     print("Alice could buy",animals&alice_ani,"from Pets R Us")
# exercise3()

# def exercise4():
#     bulbsForSale = {'daffodil': 0.35, 'tulip': 0.33, 'crocus': 0.25,'hyacinth': 0.75, 'bluebell': 0.50} 
#     mary_list={'daffodil':50,'tulip':100}
#     bulbsForSale['tulip']+=0.25*0.33
#     bulbsForSale['tulip']=round(bulbsForSale['tulip'],2)
#     mary_list['hyacinth']=30
#     sorted_list=[]
#     j,cost,quantity=0,0,0
#     for i in sorted(mary_list.keys()):
#         sorted_list.append(str.upper(i[0:3]))
#         print("{:<5s} * {:4d} =${:6.2f}".format(sorted_list[j],mary_list[i],bulbsForSale[i]*mary_list[i]))
#         cost+=bulbsForSale[i]*mary_list[i]
#         j+=1
#     for i in mary_list.values():
#         quantity+=i
#     print("You have purchased a total of",quantity,"from Bluebell Greenhouses")
#     print("Your total comes to ${:6.2f}".format(cost))

# exercise4()

import time

# def smallestIndex(data, first, last):
    
#     if first == last:
#         return first
#     if data[first] > data[last]:
#         first+=1
#     else:
#         last-=1
#     return smallestIndex(data, first, last)

# data=[8,-100,3,7]
# print(smallestIndex(data, 0, len(data)-1))
start=time.time()   

# MINIMUM= 32
  
# def find_minrun(n): 
  
#     r = 0
#     while n >= MINIMUM: 
#         r |= n & 1
#         n >>= 1
#     return n + r 
  
# def insertion_sort(array, left, right): 
#     for i in range(left+1,right+1):
#         element = array[i]
#         j = i-1
#         while element<array[j] and j>=left :
#             array[j+1] = array[j]
#             j -= 1
#         array[j+1] = element
#     return array
              
# def merge(array, l, m, r): 
  
#     array_length1= m - l + 1
#     array_length2 = r - m 
#     left = []
#     right = []
#     for i in range(0, array_length1): 
#         left.append(array[l + i]) 
#     for i in range(0, array_length2): 
#         right.append(array[m + 1 + i]) 
  
#     i=0
#     j=0
#     k=l
   
#     while j < array_length2 and  i < array_length1: 
#         if left[i] <= right[j]: 
#             array[k] = left[i] 
#             i += 1
  
#         else: 
#             array[k] = right[j] 
#             j += 1
  
#         k += 1
  
#     while i < array_length1: 
#         array[k] = left[i] 
#         k += 1
#         i += 1
  
#     while j < array_length2: 
#         array[k] = right[j] 
#         k += 1
#         j += 1
  
# def tim_sort(array): 
#     n = len(array) 
#     minrun = find_minrun(n) 
  
#     for start in range(0, n, minrun): 
#         end = min(start + minrun - 1, n - 1) 
#         insertion_sort(array, start, end) 
   
#     size = minrun 
#     while size < n: 
  
#         for left in range(0, n, 2 * size): 
  
#             mid = min(n - 1, left + size - 1) 
#             right = min((left + 2 * size - 1), (n - 1)) 
#             merge(array, left, mid, right) 
  
#         size = 2 * size 

  
  
  
# array = [-1,5,0,-3,11,9,-2,7,0] 
  
# print("Array:") 
# print(array) 
  
# tim_sort(array) 
# end=time.time()  
# print("Sorted Array:") 
# print(array)
# print(end-start)

# def smallestIndex(data, first, last):
   
#     if first == last:
#         return first
#     if data[first] > data[last]:
#         first+=1
#     else:
#         last-=1
#     return smallestIndex(data, first, last)
    


# def recursiveSelectionSort(data, length, first=0): 
    
#     if length==1:
#         return data
#     else:
#         min1=smallestIndex(data[first:],first,length-1)
#         temp=data[length-1]
#         data[length-1]=data[min1]
#         data[min1]=temp
#         return recursiveSelectionSort(data, length-1)

# array = [-1,5,0,-3,11,9,-2,7,0] 
# print(recursiveSelectionSort(array, len(array)))
# end=time.time()  

# print(end-start)
start=time.time()
array = [-1,5,0,-3,11,9,-2,7,0] 
array.sort()
end=time.time() 

print(end-start)