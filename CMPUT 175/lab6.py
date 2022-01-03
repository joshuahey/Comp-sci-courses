import matplotlib.pyplot as plt
from queues import BoundedQueue,CircularQueue
import time


def main():
    list_capacity=[]
    [list_capacity.append(i) for i in range(5000,30000,500)]
    # bqlist=[]
    # cqlist=[]
    # for m in list_capacity:
    bq=BoundedQueue(30000)
    cq=BoundedQueue(30000)
    
    # s_1=0
    # for j in range(0,5):
    for i in range(30000):
        bq.enqueue(i)
    start = time.time() 
    for i in range(30000):  
        bq.dequeue()
    end = time.time()
    time_interval = end - start
    # s_1+=time_interval
    # s_1=s_1/5
    # bqlist.append(time_interval)
    print("For Bounded Queue, the total runtime of dequeuing 50,000 items is:",time_interval)


    # s_2=0
    # # for j in range(0,5):
    for i in range(30000):
        cq.enqueue(i)
    start_2 = time.time()  
    for i in range(30000):
        cq.dequeue()
    end_2 = time.time()
    time_interval_2= end_2 - start_2
    # s_2+=time_interval
    # s_2=s_2/5
    # cqlist.append(time_interval)
    print("For Circular Queue, the total runtime of dequeuing 50,000 items is:",time_interval_2) 

    # plt.plot(list_capacity,bqlist)
    # plt.title("BQ runtime")
    # plt.show()
    # plt.plot(list_capacity,cqlist)
    # plt.title("CQ runtime")
    # plt.show()
    
main()