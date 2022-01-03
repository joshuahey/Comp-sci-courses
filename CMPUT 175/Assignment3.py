# ASSIGNMENT 3
# AUTHORS: JOSHUA GEORGE
# COLLABORATORS: NONE



import os, shlex
# the reason i import os is because my laptop cant read the files 
os.chdir(os.path.dirname(__file__))

class DLinkedListNode:
    def __init__(self, initData, initNext, initPrevious):
        self.data = initData
        self.next = initNext
        self.previous = initPrevious

        if (initPrevious != None):
            initPrevious.next = self
        if (initNext != None):
            initNext.previous = self

    def __str__(self):
        return "%s" % (self.data)

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def getPrevious(self):
        return self.previous

    def setData(self, newData):
        self.data = newData

    def setNext(self, newNext):
        self.next = newNext

    def setPrevious(self, newPrevious):
        self.previous= newPrevious

class DLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __str__(self):
        s = "[ "
        current = self.head
        while current != None:
            s += "%s " % (current)
            current = current.getNext()
        s += "]"
        return s

    def isEmpty(self):
        return self.size == 0

    def length(self):
        return self.size

    def getHead(self):
        return self.head

    def getTail(self):
        return self.tail

    def search(self, item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()
        return found

    def index(self, item):
        current = self.head
        found = False
        index = 0
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()
                index = index + 1
        if not found:
            index = -1
        return index

    def add(self, item):
        temp = DLinkedListNode(item, self.head, None)
        if self.head != None:
            self.head.setPrevious(temp)
        else:
            self.tail = temp
        self.head = temp
        self.size += 1

    def append(self, item):
        temp = DLinkedListNode(item, None, None)
        if (self.head == None):
            self.head = temp
        else:
            self.tail.setNext(temp)
            temp.setPrevious(self.tail)
        self.tail = temp
        self.size +=1

    def remove(self, item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
        if (current.getNext() != None):
            current.getNext().setPrevious(previous)
        else:
            self.tail = previous
        self.size -= 1

    def removeitem(self, current):
        previous = current.getPrevious()
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
        if (current.getNext() != None):
            current.getNext().setPrevious(previous)
        else:
            self.tail=previous
        if previous:
            self.curr = previous.getNext()
        else:
            self.curr = None
        self.size -= 1

    def insert(self, current, item, where):
        # You write this code
        # Where = 0 (before current)
        # Where = 1 (after current)
        # if where == 1 i insert the item after current else if 0 insert before current
        if where == 1:
            counter = 0
            if current == 0:
                self.add(item)
            elif current == self.size:
                self.append(item)
            else:
                current_1 = self.head
                while current_1 != None:
                    if counter == current:
                        previous = current_1.getPrevious()
                        after = DLinkedListNode(item, current_1, previous)
                        previous.setNext(after)
                        current_1.setPrevious(after)
                        self.size += 1
                    counter+=1
                    current_1=current_1.getNext()
        elif where == 0:
            counter = 0
            if current == 1:
                self.add(item)
            else:
                current_1 = self.head
                while current_1 != None:
                    if counter == current - 1:
                        previous = current_1.getPrevious()
                        after = DLinkedListNode(item, current_1, previous)
                        previous.setNext(after)
                        current_1.setPrevious(after)
                        self.size += 1
                    counter+=1
                    current_1=current_1.getNext()
                      
class TextFile:
    def __init__(self, name):
        """[name]

        Args:
            name ([str]): [textfile name]
        """
        self.__name = name
        # dlinkedlist object
        self.__content = DLinkedList()
        # current data
        self.__current = 0
        # current line/ pointer
        self.__line = 1

    def load(self, name):
        """[load file]

        Args:
            name ([str]): [name of file]
        """
        with open(name, 'r') as file:
            for line in file:
                currLine = line.strip()
                self.__content.append(currLine)
        self.setName(name)
            
    def write_1(self, name):
        """[load file]

        Args:
            name ([str]): [name of file]
        """
        file = open(name, "w")
        i = 1
        temp = self.__content.getHead()
        while i<=self.__content.length():
            if i != 1: 
                temp = temp.getNext()
            file.write(temp.getData())
            file.write("\n")
            i += 1
        self.setName(name)

    def print_1(self, offset):
        """[offset- by how much change]

        Args:
            offset ([str]): [change]
        """
        # for - operations
        prev = []
        # get current pointer place
        pointer = self.getLine()
        # for integer input
        try:
            # print current line
            if int(offset) == 0:
                start = self.getCurr()
                print(str(pointer)+":",start)
            #(+) print offset number of lines but not greater than length of list
            elif int(offset) > 0 and int(offset) < self.__content.length() :
                start = self.getCurr()  
                for i in range(int(offset)+1):
                    if start == None:
                        return
                    print(str(pointer)+":",start.getData())
                    pointer += 1
                    start = start.getNext()
            #(-) print offset number of lines  but not greater than length of list
            elif int(offset) < 0 and -int(offset) < self.getLine() :
                start = self.getCurr() 
                for i in range(-int(offset)+1):
                    if start == None:
                        return
                    prev.append(start.getData())
                    start = start.getPrevious()
                    pointer -= 1
                pointer += 1 
                for i in range(len(prev)-1, -1, -1):
                    print(str(pointer)+":",prev[i])
                    pointer +=1
            #(+) print offset number of lines greater than length of list
            elif int(offset) > self.__content.length() - self.getLine() and int(offset) > 0:
                start = self.getCurr()
                i=1
                while i<=self.__content.length():
                    if i != 1: 
                        start = start.getNext()
                    print(str(pointer)+":",start.getData())
                    if start.getNext() == None:
                        return
                    i += 1
                    pointer += 1
             #(-) print offset number of lines greater than length of list
            elif -int(offset) >= self.getLine() and int(offset) < 0 :
                start = self.getCurr()
                i=1
                while i<=self.__content.length():
                    if i != 1: 
                        prev.append(start.getData())
                        start = start.getPrevious()
                        pointer -= 1     
                    if start.getPrevious() == None:
                        prev.append(start)
                        for j in range(len(prev)-1, -1, -1):
                            print(str(pointer)+":", prev[j])
                            pointer += 1 
                        return
                    i += 1
        except ValueError:
            print("Not integer")

    def linenum(self, lineno):
        """[set line with pointer]

        Args:
            lineno ([str]): [lineno]
        """
        try:
            # get current line number
            pointer = self.getLine()  
            # Null command
            if len(lineno) == 0:
                start = self.getCurr()
                if start.getNext() == None:
                    return
                start = start.getNext()
                pointer += 1
                print(pointer,":",start.getData())
                self.setLine(pointer)
                self.setCurr(start)
                return
            # start
            elif int(lineno) == 1:
                start = self.__content.getHead()
                self.setCurr(start)
                pointer = 1
                self.setLine(1)
                return
            else :
                start = self.getCurr()
            #(+) iterate line and number 
            if pointer <= int(lineno):
                while pointer < int(lineno):
                    try:
                        start = start.getNext()
                        pointer += 1
                    except AttributeError:
                        print("Out of bounds")
                        return
            #(-) iterate line and number 
            else:
                while pointer > int(lineno):
                    try:
                        start = start.getPrevious()
                        pointer -=1 
                    except AttributeError:
                        print("Out of bounds")
                        return
            # set it
            self.setLine(pointer)
            self.setCurr(start)
        except ValueError:
            print("Not integer")
           
    def add(self, where):
        """[add/insert]

        Args:
            where ([str]): [where 0/1]
        """
        
        add = []
        #  add/insert to empty file
        with open(self.getName(), "r+") as file:
            first = file.read()
            g = True
            if not first:
                while g:
                    choice = input("")
                    if choice == "":
                        g = False
                    file.write(choice)
                    file.write("\n")
                return
        #  get curr line and number
        pointer = self.getLine()
        start = self.getCurr()
        g = True
        # add/insert
        while g:
            choice = input("")
            if choice == "":
                g = False
            add.append(choice)
        for i in range(len(add)-2, -1, -1):
            self.__content.insert(pointer, add[i], where)
        #(1) set to currline and number
        if where == 1:
            for i in range(len(add)-1):
                start =  start.getNext()
                pointer +=1
            self.setCurr(start)
            self.setLine(pointer)
        #(0) set to currline and number
        else:
            for i in range(len(add)-1):
                start =  start.getPrevious()
                pointer +=1
            self.setCurr(start)
            self.setLine(pointer)

    def delete(self, offset):
        """[delete]

        Args:
            offset ([str]): [offset]
        """
        # empty file error
        with open(self.getName(), "r+") as file:
            first = file.read()
            if not first:
                file.close()
                print("Error empty file")
                return
        # get current line and number
        start = self.getCurr()
        pointer = self.getLine() 
        # delete current line
        if int(offset) == 0:
            next = start.getNext()
            self.__content.removeitem(start)
            start = next
            self.setCurr(start)
        #(+) delete offset number of lines 
        elif int(offset) > 0:
            for i in range(int(offset)+1):
                curr = start.getNext()
                self.__content.removeitem(start)
                if curr == None:
                    self.setCurr(self.__content.getTail())
                    return
                start = curr
            self.setCurr(start)
        #(-) delete offset number of lines 
        elif int(offset) < 0:
            for i in range(-int(offset)+1):
                prev = start.getPrevious()
                self.__content.removeitem(start)
                start = prev
            for i in range(-int(offset)+1):
                start = start.getNext()
                if start == None:
                    self.setCurr(self.__content.getTail())
                    return
            self.setCurr(start)
    
    def search(self, text, where):
        """[search text]

        Args:
            text ([str]): [text to be found]
            where ([int]): [forward or backward]
        """
        # get current line and number
        start = self.getCurr()
        pointer = self.getLine()
        check = 0
        # (+) direction
        if where == 1:
            found = False
            while not found :       
                if pointer > self.__content.length():
                    start = self.__content.getHead()
                    temp = start.getData()               
                    pointer = 1
                else:
                    temp = start.getData()
                if text in temp:
                    print(str(pointer)+":","".join(temp))
                    found = True
                    return
                start = start.getNext()
                pointer +=1
                check +=1
                if check > self.__content.length():
                    print("Not found")
                    return

            self.setCurr(start)
            self.setLine(pointer)
         # (-) direction
        elif where == 0:
            found = False
            while not found :
                if pointer <= 0:
                    start = self.__content.getTail()
                    temp = start.getData()
                    pointer = self.__content.length()-1
                else:
                    temp = start.getData()
                if text in temp:
                    print(str(pointer)+":","".join(temp))
                    found = True
                    return              
                start = start.getPrevious()
                pointer -=1
                check -=1
                if -check > self.__content.length():
                    print("Not found")
                    return

            self.setCurr(start)
            self.setLine(pointer)

    def replace(self, text1, text2):
        """[replace text1 with text2]

        Args:
            text1 ([str]): [text1]
            text2 ([str]): [tex2]
        """
        # in empty file
        with open(self.getName(), "r+") as file:
            first = file.read()
            if not first:
                file.close()
                return

        start = self.getCurr()
        pointer = self.getLine()
        # replace text
        if len(text2) !=0:    
            temp = start.getData().replace(text1, text2)
            start.setData(temp)
            print(str(pointer)+":",start)
            self.setCurr(start)
        else:
            temp = start.getData().replace(text1, "")
            start.setData(temp)
            print(str(pointer)+":",start)
            self.setCurr(start)       

    def sort(self):
        """[sort]
        # my sort function prints the sorted list as well 
        Returns:
            [sorted data]: [sort]
        """
        
        # in empty file
        with open(self.getName(), "r+") as file:
            first = file.read()
            if not first:
                file.close()
                return
        # get data
        unsorted = []
        i = 1
        temp = self.__content.getHead()
        unsorted.append(temp.getData())
        while i<=self.__content.length():
            if i != 1: 
                temp = temp.getNext()
                unsorted.append(temp.getData())
            i += 1
        # function to make calls easier (ie for lower and capital sentences separation)
        def selectionsort(data):
            for index in range(len(data)-1):
                smallIndex = index
                for i in range(index+1,len(data)):
                    if (data[i]<data[smallIndex]):
                        smallIndex=i
            
                temp=data[index] 
                data[index]=data[smallIndex]
                data[smallIndex]=temp
            return data
        # sorting
        caps = []
        lower = []
        for i in unsorted:
            if i[0].isupper():
                caps.append(i)
            if i[0].islower():
                lower.append(i)
        selectionsort(caps)
        selectionsort(lower)

        data = caps + lower 
        # replacing dlinkedlist text with new sorted text
        start = self.__content.getHead()
        i = 1
        while i <= self.__content.length():  
            self.setCurr(start)
            self.replace(start.getData(), data[i-1])
            start =  start.getNext()
            i += 1

        self.getCurr()
        
    def getName(self):
        """[get file name]

        Returns:
            [str]: [returns file name]
        """
        return self.__name

    def setName(self, name):
        """[set file name]

        Returns:
            [str]: [sets file name]
        """
        self.__name = name

    def getCurr(self): 
        """[get current line]

        Returns:
            [str]: [returns current line]
        """
        return self.__current

    def setCurr(self, current):
        """[set current line]

        Returns:
            [str]: [sets current line]
        """
        self.__current = current

    def getLine(self):
        """[get current line number]

        Returns:
            [str]: [returns current line number]
        """
        return self.__line

    def setLine(self, line):
        """[set current line number]

        Returns:
            [str]: [sets current line number]
        """
        self.__line = line
    
def main():

    quit = False
    test = None
    while not quit:

        inp_command = input(">")
        inp_command = shlex.split(inp_command)

        if inp_command != []:
            if inp_command[0] =='q':
                quit = True

            elif inp_command[0] == 'l':
                try:
                    with open(inp_command[1],"r+") as file:
                        test = TextFile(inp_command[1])
                        test.load(inp_command[1])
                        store = inp_command[1]
                except OSError:
                        print("No file exists")

            elif inp_command[0] == "w":
                if len(inp_command) == 1:
                    test.write_1(store)
                else:
                    try:
                        with open(inp_command[1], "x") as file_2:
                            test = TextFile(inp_command[1])
                            test.write_1(inp_command[1])
                    except FileExistsError:
                        print("File exists")

            elif inp_command[0] == 'p':
                test.print_1(0) if len(inp_command) ==  1 else  test.print_1(inp_command[1])

            elif inp_command[0].isdigit():
                test.linenum(inp_command[0])
            
            elif inp_command[0] == " ":
                print(test.getCurr())
                
            elif inp_command[0] == 'a':
                test.add(1) 

            elif inp_command[0] == 'i':
                test.add(0)

            elif inp_command[0] == 'd':
                test.delete(0) if len(inp_command) ==  1 else  test.delete(inp_command[1])
            
            elif inp_command[0] == '/':
                test.search(inp_command[1],1)

            elif inp_command[0] == '?':
                test.search(inp_command[1],0)

            elif inp_command[0] == 'r':
                test.replace(inp_command[1],"") if len(inp_command) == 2 else test.replace(inp_command[1],inp_command[2])
            
            elif inp_command[0] == 's':
                test.sort()
        else:
            test.linenum("")

main()