
def all_the_same(my_list):
   same = True
   first_number = my_list[0]
   index = 0
   while index < len(my_list) and same:
      if my_list[index]!= first_number: 
         same = False
         print("hey")
      index = index + 1
   return same
a=[1,1,1,2]
all_the_same(a)      
   

