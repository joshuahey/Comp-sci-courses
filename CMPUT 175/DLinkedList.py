#----------------------------------------------------
# Lab 7, Exercise 2: Doubly Linked Lists
# TO DO: complete mandatory methods in DLinkedList class
# TO DO (optional): complete optional methods in DLinkedList class
# to get better understanding of manipulating linked lists
#
# Author: 
# Collaborators/references:
#   - CMPUT 175 provided complete DLinkedListNode
#   - CMPUT 175 provided init, search, index, traverseBackwards methods for DLinkedList
#   - CMPUT 175 provided tests for DLinkedList
#----------------------------------------------------


class DLinkedListNode:
    # An instance of this class represents a node in Doubly-Linked List
    def __init__(self,initData,initNext,initPrevious):
        self.data = initData
        self.next = initNext
        self.previous = initPrevious
        
        if initNext != None:
            self.next.previous = self
        if initPrevious != None:
            self.previous.next = self
            
    def getData(self):
        return self.data
    
    def setData(self,newData):
        self.data = newData
        
    def getNext(self):
        return self.next
    
    def getPrevious(self):
        return self.previous
    
    def setNext(self,newNext):
        self.next = newNext
        
    def setPrevious(self,newPrevious):
        self.previous = newPrevious

class DLinkedList:
    # An instance of this class represents the Doubly-Linked List
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__size = 0        
           
    def search(self, item):
        current = self.__head
        found = False
        while current != None and not found:
            if current.getData() == item:
                found= True
            else:
                current = current.getNext()
        return found
        
    def index(self, item):
        current = self.__head
        found = False
        index = 0
        while current != None and not found:
            if current.getData() == item:
                found= True
            else:
                current = current.getNext()
                index = index + 1
        if not found:
                index = -1
        return index        
    
    def traverseBackwards(self):
        backwardString = ''
        current = self.__tail
        while current != None:
            backwardString += str(current.getData())
            current = current.getPrevious()
        return backwardString         
        
    def insert(self, pos, item): 
        assert isinstance(pos, int), ('Error: Type error: {}'.format(type(pos)))
        assert pos >= 0, ('Error: not positive {}'.format(pos))
        counter=0
        if pos==0:
            self.add(item)
        elif pos==self.__size:
            self.append(item)
        else:
            current=self.__head
            while current!=None:
                if counter==pos:
                    previous = current.getPrevious()
                    after = DLinkedListNode(item, current, previous)
                    previous.setNext(after)
                    current.setPrevious(after)
                    self.__size += 1
                   
                counter+=1
                current=current.getNext()
        

    def searchLarger(self, item):
        current=self.__head
        position=0
        found=False
        while current!=0 and not found:
            if current.getData()>item:
                found=True
            else:
                current=current.getNext()
                position+=1
        if found==False:
            position=-1
        return position
        
        
    def getSize(self):
        return self.__size
    
    def getItem(self, pos):      
        if pos> self.__size:
            raise Exception("out of range")
        if not isinstance(pos, int):
            raise Exception("Not int")
        if pos <0:
            pos+=self.__size
        counter=0
        current=self.__head
        while current!=None:
            if counter==pos:
                return current.getData()
            counter+=1
            current=current.getNext()
        
    def __str__(self):    
        
        current = self.__head
        list1 = []
        while current!=None:
            list1.append(str(current.getData()))
            current = current.getNext()
        return " ".join(list1)


    def add(self, item):
        new_node = DLinkedListNode(item, self.__head, None)  
        if self.__head != None: 
            self.__head.setPrevious(new_node)
        else:
            self.__tail=new_node
        self.__head = new_node
        self.__size += 1

    def remove(self, item):
        pass
        
    def append(self, item):
        new_node=DLinkedListNode(item,None,None)
        if self.__head == None:
            self.__head = new_node
        else:
            self.__tail.setNext(new_node)
            new_node.setPrevious(self.__tail)     
        self.__tail = new_node
        self.__size += 1
     
    def pop1(self):
        # optional exercise
        pass
    
    def pop(self, pos=None):
        # optional exercise
        # Hint - incorporate pop1 when no pos argument is given
        pass
        
    

def test():
                  
    linked_list = DLinkedList()
                    
    is_pass = (linked_list.getSize() == 0)
    assert is_pass == True, "fail the test"

    linked_list.insert(0, "Hello")
    linked_list.insert(1, "World")    
    
    is_pass = (str(linked_list) == "Hello World")
    assert is_pass == True, "fail the test"
    
    is_pass = (linked_list.traverseBackwards() == "WorldHello")
    assert is_pass == True, "fail the test"
              
    is_pass = (linked_list.getSize() == 2)
    assert is_pass == True, "fail the test"
            
    is_pass = (linked_list.getItem(0) == "Hello")
    assert is_pass == True, "fail the test"
        
        
    is_pass = (linked_list.getItem(1) == "World")
    assert is_pass == True, "fail the test"    
            
    is_pass = (linked_list.getItem(0) == "Hello" and linked_list.getSize() == 2)
    assert is_pass == True, "fail the test"

    '''
    OPTIONAL TESTS FOR OPTIONAL EXERCISE - do not need to demo
    '''
    '''
    is_pass = (linked_list.pop(1) == "World")
    assert is_pass == True, "fail the test" 
    
    is_pass = (linked_list.traverseBackwards() == "Hello")
    assert is_pass == True, "fail the test"
            
    is_pass = (linked_list.pop() == "Hello")
    assert is_pass == True, "fail the test"     
            
    is_pass = (linked_list.getSize() == 0)
    assert is_pass == True, "fail the test" 
    
    int_list2 = DLinkedList()
                    
    for i in range(0, 10):
        int_list2.add(i)
    
    is_pass = (str(int_list2) == "9 8 7 6 5 4 3 2 1 0")
    assert is_pass == True, "fail the test"
        
    is_pass = (int_list2.traverseBackwards() == "0123456789")
    assert is_pass == True, "fail the test"
    
    int_list2.remove(1)
    int_list2.remove(3)
    int_list2.remove(2)
    int_list2.remove(0)
    
    is_pass = (str(int_list2) == "9 8 7 6 5 4")
    assert is_pass == True, "fail the test"
    
    is_pass = (int_list2.traverseBackwards() == "456789")
    assert is_pass == True, "fail the test"
                
    for i in range(11, 13):
        int_list2.append(i)
    is_pass = (str(int_list2) == "9 8 7 6 5 4 11 12")
    assert is_pass == True, "fail the test"
    
    is_pass = (int_list2.traverseBackwards() == "1211456789")
    assert is_pass == True, "fail the test"
                
    for i in range(21, 23):
        int_list2.insert(0,i)
    is_pass = (str(int_list2) == "22 21 9 8 7 6 5 4 11 12")
    assert is_pass == True, "fail the test"
                
    is_pass = (int_list2.getSize() == 10)
    assert is_pass == True, "fail the test"    
    '''
                    
    int_list = DLinkedList()
                    
    is_pass = (int_list.getSize() == 0)
    assert is_pass == True, "fail the test"                   
                    
    for i in range(9, -1, -1):
        int_list.insert(0,i)      
            
    is_pass = (int_list.getSize() == 10)
    assert is_pass == True, "fail the test"            
            
    is_pass = (int_list.searchLarger(8) == 9)
    assert is_pass == True, "fail the test"
            
    int_list.insert(7,801)   
        
    is_pass = (int_list.searchLarger(800) == 7)
    assert is_pass == True, "fail the test"
                  
    is_pass = (int_list.getItem(-1) == 9)
    assert is_pass == True, "fail the test"
            
    is_pass = (int_list.getItem(-4) == 801)
    assert is_pass == True, "fail the test"
                   
    if is_pass == True:
        print ("=========== Congratulations! Your have finished exercise 2! ============")
            
if __name__ == '__main__':
    test()
