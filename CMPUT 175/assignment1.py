# ASSIGNMENT 1
# AUTHORS: JOSHUA GEORGE
# COLLABORATORS: NONE




import os
os.chdir(os.path.dirname(__file__))

# Initialise student details dictionary containing all students and their courses
# Initialise dict courses to store info from courses.txt
# Initilise dict enroll to store info from enrollment.txt

student_details={} 
dict_courses={}
dict_enroll={}

# open file and maap courses with their time day and capacity
with open("courses.txt","r+") as f1:
    for line in f1:
        currentline=line.strip().split("; ")
        dict_courses[currentline[0]]=[currentline[1],currentline[2],currentline[3]]
    f1.close()

# open file and get map courses to student id
with open("enrollment.txt","r+") as f2:
    for line in f2:  
        currentline2=line.strip().split(": ")
        if currentline2[1] in dict_enroll:
            dict_enroll[currentline2[1]].append(currentline2[0])
        else:
            dict_enroll[currentline2[1]]=[currentline2[0]]
    f2.close()

# Complete student detaisl with the format {"student id": [course stuff],...}
for k,v in dict_enroll.items():
        b=[]
        for j in v:
            for key,value in dict_courses.items():
                if j == key:
                    b.append([j,value[0],value[1]])               
        student_details[k]=b

def student(student_id):
    """[check validity of student and get student details]
    Args:
        student_id ([string]): [student_id]
    Returns:
        [dictionary]: [dict with student name and faculty]
    """
    dict_student={}
    with open("students.txt","r+") as f:
        for line in f:
            currentline=line.strip().split(",")
            dict_student[currentline[2]]=[currentline[0],currentline[1]]

    dict_values=list(dict_student.values())
    a=list()
    for i in range(len(dict_values)):
        a.append(dict_values[i][0])

    # check if invalid student id
    if student_id not in a:
        print("Invalid Student ID. Cannot print timetable")    
        main()
    return dict_student
        
def timetable(dict_student,student_id):
    """[print timetable]
    Args:
        dict_student ([dictionary]): [from student()]
        student_id ([string]): [student_id]
    Returns:
        [dict_courses]: [all courses]
    """
    for key, value in dict_student.items():
        if student_id==value[0]:
            print("Time table for",key.upper(),"in the faculty of",value[1])
    for k in student_details[student_id]:
        k[1]=k[1].split(" ")
        k[0]=k[0].split(" ")

    # print timetable section   
    days=["Mon","Tues","Wed","Thurs","Fri"]
    for k in days:
        print("      ",k.center(4," "),end='')
    print("")
    print("     ",5*("+"+"-"*9)+"+")
    timings=[" 8:00"," 8:30"," 9:00"," 9:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00","16:30"]
    j=-1
    line1=[5,11,23,29,41,47]
    line2=[8,26,44]
    line3=[17,35,53]

    for i in range(0,54):
         # if else conditionals for printing timetable
        if i in line1 and i!=0:
            print("     ","|"+"---------"+"|","       ","|"+"---------"+"|","       ","|"+"---------"+"|")
        elif i%6==0 :
            a,b,c=[],[],[]
            j=j+1 
            print(timings[j],"|","       ","|","       ","|","       ","|","       ","|","       ","|")
            for k in student_details[student_id]:
                 # this works like a coordinate system to input the courses
                if timings[j].strip()==k[1][1] and "MWF"== k[1][0]:
                    # truncating len word>4 stuff
                    if len(k[0][0])>4:
                        t=k[0][0][0]+k[0][0][1]+k[0][0][2]+"*"+" "+k[0][1]
                        a.append("".join(t))    
                    else:
                        a.append(" ".join(k[0]))
                    # to align the number of student values properly
                    if len(k[2])<3 and len(k[2])!=1:
                        p=str(k[2])
                        i=" "+p[0]+p[1] 
                        c.append(i)
                    elif len(k[2])<2:
                        p=str(k[2])
                        i=" "+p[0]+" " 
                        c.append(i)
                    else:
                        c.append(k[2])   
                    print("      |{r} |         |{r} |         |{r} |".format(r=a[0]))
                    print("      |   {r}   |         |   {r}   |         |   {r}   |".format(r=c[0]))                  
                elif timings[j].strip()==k[1][1] and "TR"== k[1][0]:
                    # truncating len word>4 stuff
                    if len(k[0][0])>4:
                        t=k[0][0][0]+k[0][0][1]+k[0][0][2]+"*"+" "+k[0][1]
                        b.append("".join(t))
                    else:
                        b.append(" ".join(k[0]))
                    # to align the number of student values properly
                    if len(k[2])<3 and len(k[2])!=1:
                        p=str(k[2])
                        i=" "+p[0]+p[1] 
                        c.append(i)
                    elif len(k[2])<2:
                        p=str(k[2])
                        i=" "+p[0]+" " 
                        c.append(i)
                    else:
                        c.append(k[2]) 
                    print("      |         |{r} |         |{r} |         |" .format(r=b[0]))
                    print("      |         |   {r}   |         |   {r}   |         |".format(r=c[0]))    
        elif i%3==0 :
            d,e=[],[]
            j=j+1
            print(timings[j],"|","       ","|","       ","|","       ","|","       ","|","       ","|")
            for k in student_details[student_id]:
                # this works like a coordinate system to input the courses
                if timings[j].strip()==k[1][1] and "TR"== k[1][0]:
                    # truncating len word>4 stuff
                    if len(k[0][0])>4:
                        z=k[0][0][0]+k[0][0][1]+k[0][0][2]+"*"+" "+k[0][1]
                        d.append("".join(z))
                    else:
                        d.append(" ".join(k[0]))
                    # to align the number of student values properly
                    if len(k[2])<3 and len(k[2])!=1:
                        p=str(k[2])
                        i=" "+p[0]+p[1] 
                        e.append(i)
                    elif len(k[2])<2:
                        p=str(k[2])
                        i=" "+p[0]+" " 
                        e.append(i)
                    else:
                        e.append(k[2]) 
                    print("      |         |{r} |         |{r} |         |" .format(r=d[0]))
                    print("      |         |   {r}   |         |   {r}   |         |".format(r=e[0]))
        elif i in line2:
            print("     ","|","       ","|"+"---------"+"|","       ","|"+"---------"+"|","       ","|") 
        elif i in line3:
            print("     ",5*("+"+"-"*9)+"+") 
        else:
            print("     ","|","       ","|","       ","|","       ","|","       ","|","       ","|")
    # split the for example [CMPUT 175] list before so now join and also [TWF 9:00]  
    for k in student_details[student_id]:
        k[1]=" ".join(k[1])
        k[0]=" ".join(k[0]) 
    return student_details,dict_courses

def enroll(dict_student,student_id):
    """[enrolling in courses]
    Args:
        dict_student ([dictionary]): [student details]
        student_id ([string]): [student id]
    Returns:
        [list]: [list of enrolled courses]
    """
    course_id=input("enter course name:")
    # invalid course
    if course_id not in dict_courses:
        print("Invalid course")
        main()
    # check if course exists or if the time clashes
    for i in student_details[student_id]:
        if course_id in dict_enroll[student_id] or dict_courses[course_id][0] in i :
            print("Schedule conflict: Student already in course on",dict_courses[course_id][0])
            main()
     # check if course capacity is full
    if dict_courses[course_id][1]=="0":
        print(course_id,"is already at capacity. Please contact advisor to get on waiting list.")
        main()
    # add number of students in the course
    dict_courses[course_id][1]=int(dict_courses[course_id][1])
    dict_courses[course_id][1]-=1
    # append the course
    for key,value in dict_courses.items():
        if course_id==key:
            student_details[student_id].append([course_id,value[0],value[1]])
    for key, value in dict_student.items():
        if student_id==value[0]:
            print(key,"has succesully enrolled in",course_id,"on",dict_courses[course_id][0])
    return student_details,dict_courses,[course_id,student_id]

def drop(dict_student,student_id):
    """[drop courses]
    Args:
        dict_student ([dictionary]): [student details]
        student_id ([string]): [student id]
    Returns:
        [list]: [list of dropped courses]
    """
    #  REFERENCE:https://www.learnpythonwithrune.org/sort-a-python-list-with-string-of-integers-or-a-mixture
    # sort the array by appending to a list
    p=[]
    def comp(o):
        return o.split()[0]
    for i in student_details[student_id]:
        p.append(i[0])
    p.sort(key=comp)
    # print the courses
    print("Select course to drop")
    for i in p:
        print("-",i)
    course_delete=input(">")
    # check validity
    if course_delete not in p:
        print("Invalid course")
        main()  
    # subtract number of students in the course
    dict_courses[course_delete][1]=int(dict_courses[course_delete][1])
    dict_courses[course_delete][1]+=1
    dict_courses[course_delete][1]=str(dict_courses[course_delete][1])
    # pop the course from students courses
    for i in student_details[student_id]:
        for j in i:
            if course_delete==j:
                student_details[student_id].remove(i)

    for key, value in dict_student.items():
        if student_id==value[0]:
            print(key,"has succesully dropped",course_delete)  
    return student_details,dict_courses,[course_delete,student_id]

def quit1(en,dr):
    """[quit the program]
s:
        en ([list]): [list of enrolled courses of all students]
        dr ([list]): [list of dropped courses of all students]

    Returns:
        [type]: [description]
    """
    # write to file
    with open("enrollment.txt","a") as f3:  
        f3.write("\n")
        for i in en:
            f3.write(": ".join(i))
            f3.write("\n")
        for j in dr:
            f3.write(": ".join(j))
            f3.write("\n")
    f3.close()
    return "Goodbye"

def main():
    all_students_enroll=[]
    all_students_drop=[]
    print("="*26)
    print("Welcome to Mini-BearTracks")
    print("="*26)
    # menu
    while 1:
        print("What would you like to do?")
        valid={1,2,3,4}
        print("1. Print timetable ")
        print("2. Enroll in course")
        print("3. Drop course")     
        print("4. Quit")
        choice=int(input(">")) 
        if choice in valid:
            if choice== 1:
                student_id=input("Student ID:")
                timetable(student(student_id),student_id)
            elif choice ==2:
                student_id=input("Student ID:")
                result1=enroll(student(student_id),student_id)  
                all_students_enroll.append(result1[2])
            elif choice ==3 :
                student_id=input("Student ID:")
                result2=drop(student(student_id),student_id) 
                all_students_drop.append(result2[2])  
            elif choice==4:
                quit1(all_students_enroll,all_students_drop)
                quit()
        else:
            print("Sorry, invalid entry. Please enter a choice from 1 to 4.")

main()

    